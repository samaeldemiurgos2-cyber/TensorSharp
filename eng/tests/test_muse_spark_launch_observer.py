"""An intended launch must not masquerade as observed matching execution."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / 'validation'
SPEC = importlib.util.spec_from_file_location('observe_spark_llama', ROOT / 'observe-muse-spark-llama.py')
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)
TS_SPEC = importlib.util.spec_from_file_location('observe_spark_tensorsharp', ROOT / 'observe-muse-spark-tensorsharp.py')
tensorsharp = importlib.util.module_from_spec(TS_SPEC)
TS_SPEC.loader.exec_module(tensorsharp)


class MuseSparkLaunchObserverTests(unittest.TestCase):
    def profile(self, dtype='f16'):
        return {'settings': {'layer_ranges': [[0,34],[35,41],[42,48],[49,55],[56,61]],
            'cpu_expert_layer_count':32,'cpu_threads':32,'key_cache_dtype':dtype,'value_cache_dtype':dtype,
            'max_running_requests':4,'aggregate_kv_pool_tokens':16384,'context_per_request':16384,
            'max_batched_tokens':256,'prefill_chunk_tokens':256,'max_output_tokens':512,
            'prefix_cache':True,'speculation':'disabled'},
            'server_url':'http://127.0.0.1:8082',
            'weights_sha256':{'UD-Q3_K_XL/spark-00001-of-00010.gguf':'a'*64},
            'tokenizer_metadata':{'file':'/workspace/models/muse-spark/UD-Q3_K_XL/spark-00001-of-00010.gguf'},
            'reference_prompt_adapter':{'template':'/workspace/template.jinja'}}

    def argv(self):
        pairs={'--n-cpu-moe':'32','--threads':'32','--threads-batch':'32',
            '--cache-type-k':'f16','--cache-type-v':'f16','--parallel':'4','--ctx-size':'16384',
            '--kv-unified-per-slot':'16384','--batch-size':'256','--ubatch-size':'256',
            '--predict':'512','--tensor-split':'35,7,7,7,7','--device':'CUDA0,CUDA1,CUDA2,CUDA3,CUDA4',
            '--gpu-layers':'all','--split-mode':'layer','--mmproj-device':'CUDA0',
            '--override-tensor':r'token_embd\.weight=CUDA0','--fit':'off','--flash-attn':'on',
            '--reasoning':'off','--host':'127.0.0.1','--port':'8082',
            '--model':'/workspace/models/muse-spark/UD-Q3_K_XL/spark-00001-of-00010.gguf',
            '--mmproj':'/workspace/models/muse-spark/mmproj-Q8_0.gguf',
            '--chat-template-file':'/workspace/template.jinja'}
        return ['build/bin/llama-server']+[v for pair in pairs.items() for v in pair]+[
            '--kv-unified','--cont-batching','--cache-prompt','--jinja']

    def test_complete_profile_includes_output_layer_on_last_gpu(self):
        checked=observer.verify_arguments(self.argv(),self.profile())
        self.assertEqual('35,7,7,7,7',checked['--tensor-split'])
        self.assertEqual(16384,checked['--ctx-size'])

    def test_wrong_precision_pool_placement_port_or_weight_path_is_rejected(self):
        for flag,wrong in [('--cache-type-k','f32'),('--ctx-size','65536'),
                           ('--tensor-split','35,7,7,7,6'),('--port','8083'),
                           ('--model','/other/UD-Q3_K_XL/spark-00001-of-00010.gguf'),
                           ('--mmproj','/other/mmproj-Q8_0.gguf')]:
            with self.subTest(flag=flag):
                argv=self.argv();argv[argv.index(flag)+1]=wrong
                with self.assertRaises(ValueError):observer.verify_arguments(argv,self.profile())

    def test_missing_repeated_and_alias_overrides_are_rejected(self):
        for tail in [['--ctx-size','8192'],['-c','8192'],['--no-cache-prompt']]:
            with self.subTest(tail=tail):
                with self.assertRaises(ValueError):observer.verify_arguments(self.argv()+tail,self.profile())
        argv=self.argv();argv.remove('--cont-batching')
        with self.assertRaises(ValueError):observer.verify_arguments(argv,self.profile())

    def test_new_profile_requires_exact_new_batch_and_dtype(self):
        profile=self.profile('f32');profile['settings']['max_batched_tokens']=128
        profile['settings']['prefill_chunk_tokens']=128
        with self.assertRaises(ValueError):observer.verify_arguments(self.argv(),profile)
        argv=self.argv()
        for flag,value in [('--cache-type-k','f32'),('--cache-type-v','f32'),('--batch-size','128'),('--ubatch-size','128')]:
            argv[argv.index(flag)+1]=value
        observer.verify_arguments(argv,profile)

    def test_llama_final_cache_flag_must_match_profile(self):
        profile=self.profile();profile['settings']['prefix_cache']=False
        profile['engine_options']={'prompt_cache_ram_mib':0,'cache_idle_slots':False}
        with self.assertRaises(ValueError):observer.verify_arguments(self.argv(),profile)
        tail=['--no-cache-prompt','--cache-ram','0','--no-cache-idle-slots']
        observer.verify_arguments(self.argv()+tail,profile)
        with self.assertRaises(ValueError):
            observer.verify_arguments(self.argv()+tail+['--cache-prompt'],profile)

    def test_llama_no_cache_must_also_disable_ram_and_idle_slot_state_copies(self):
        profile=self.profile();profile['settings']['prefix_cache']=False
        profile['engine_options']={'prompt_cache_ram_mib':0,'cache_idle_slots':False}
        tail=['--no-cache-prompt','--cache-ram','0','--no-cache-idle-slots']
        checked=observer.verify_arguments(self.argv()+tail,profile)
        self.assertEqual(0,checked['--cache-ram'])
        self.assertTrue(checked['--no-cache-idle-slots'])
        for bad_tail in [tail[:-1], ['--no-cache-prompt','--cache-ram','8192','--no-cache-idle-slots'],
                         tail+['--cache-idle-slots'], tail+['--cache-ram','8192']]:
            with self.subTest(tail=bad_tail),self.assertRaises(ValueError):
                observer.verify_arguments(self.argv()+bad_tail,profile)
        for options in ({}, {'prompt_cache_ram_mib':8192,'cache_idle_slots':False},
                        {'prompt_cache_ram_mib':0,'cache_idle_slots':True},
                        {'prompt_cache_ram_mib':False,'cache_idle_slots':False}):
            profile['engine_options']=options
            with self.subTest(options=options),self.assertRaises(ValueError):
                observer.verify_arguments(self.argv()+tail,profile)

    def test_llama_cache_enabled_profile_still_accepts_ram_cache(self):
        checked=observer.verify_arguments(self.argv()+['--cache-ram','8192','--cache-idle-slots'],self.profile())
        self.assertEqual('35,7,7,7,7',checked['--tensor-split'])

    def ts_profile(self):
        planned=self.profile();planned['server_url']='http://127.0.0.1:8083'
        planned['settings']['prefix_cache']=False
        options={'TS_MUSE_SPARK_STREAM_EXPERTS':'1','TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES':'1',
                 'TS_MUSE_SPARK_FUSE_RESIDUALS':'1','DOTNET_PROCESSOR_COUNT':'40','TS_CODE_EXEC':'0'}
        planned['engine_options']={'environment':options}
        argv=['dotnet','TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll',
              '--model',planned['tokenizer_metadata']['file'],
              '--mmproj','/workspace/models/muse-spark/mmproj-Q8_0.gguf',
              '--backend','ggml_cuda','--layer-split','5','--n-cpu-moe','32',
              '--cpu-moe-threads','32','--kv-cache-dtype','f16','--host','127.0.0.1','--port','8083',
              '--max-tokens','512','--sampling-precedence','request','--no-spec','--no-webui','--no-prefix-cache']
        env={'MAX_CONTEXT':'16384','TS_SCHED_NUM_BLOCKS':'1024','TS_SCHED_MAX_RUNNING_SEQS':'4',
             'TS_SCHED_MAX_BATCHED_TOKENS':'256','TS_MUSE_SPARK_PREFILL_CHUNK':'256',
             'TS_SCHED_PREFILL_CHUNK':'256','TS_SCHED_SOLO_PREFILL_CHUNK':'256',
             'TS_MUSE_SPARK_LAYER_SPLIT':'35,7,7,7,6','TS_CPU_MOE_THREADS':'32',
             'TS_GGML_CPU_THREADS':'32','OMP_NUM_THREADS':'32',**options}
        return planned,argv,env

    def test_tensorsharp_exact_common_and_engine_specific_options(self):
        planned,argv,env=self.ts_profile()
        arguments,checked_env=tensorsharp.verify_profile(argv,env,planned)
        self.assertEqual(32,arguments['--n-cpu-moe'])
        self.assertEqual('1',checked_env['TS_MUSE_SPARK_STREAM_EXPERTS'])
        self.assertNotIn('environment',planned['settings'])

    def test_tensorsharp_missing_wrong_optimization_and_common_values_fail(self):
        for key in ('TS_MUSE_SPARK_STREAM_EXPERTS','TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES',
                    'TS_MUSE_SPARK_FUSE_RESIDUALS','TS_MUSE_SPARK_LAYER_SPLIT','TS_SCHED_NUM_BLOCKS'):
            for value in (None,'unexpected'):
                with self.subTest(key=key,value=value):
                    planned,argv,env=self.ts_profile()
                    if value is None:env.pop(key)
                    else:env[key]=value
                    with self.assertRaises(ValueError):tensorsharp.verify_profile(argv,env,planned)
        planned,argv,env=self.ts_profile()
        planned['engine_options']['environment'].pop('TS_MUSE_SPARK_STREAM_EXPERTS')
        with self.assertRaises(ValueError):tensorsharp.verify_profile(argv,env,planned)

    def test_tensorsharp_cache_speculation_and_conflicting_environment_fail(self):
        for removed in ('--no-prefix-cache','--no-spec'):
            planned,argv,env=self.ts_profile();argv.remove(removed)
            with self.assertRaises(ValueError):tensorsharp.verify_profile(argv,env,planned)
        planned,argv,env=self.ts_profile()
        planned['engine_options']['environment']['MAX_CONTEXT']='32768'
        with self.assertRaises(ValueError):tensorsharp.verify_profile(argv,env,planned)
        planned,argv,env=self.ts_profile();env['TS_SCHED_BLOCK_SIZE']='32'
        with self.assertRaises(ValueError):tensorsharp.verify_profile(argv,env,planned)

    def test_tensorsharp_no_spec_rejects_later_case_insensitive_enable_and_inherited_draft(self):
        for tail in (['--spec'], ['--SPEC']):
            planned, argv, env = self.ts_profile()
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                tensorsharp.verify_profile(argv + tail, env, planned)
        for key, value in (('TS_SPEC', '1'), ('TS_SPEC', 'true'),
                           ('TS_SPEC_DRAFT_MODEL', '/model/draft.gguf')):
            planned, argv, env = self.ts_profile()
            env[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                tensorsharp.verify_profile(argv, env, planned)
        planned, argv, env = self.ts_profile()
        env['TS_SPEC'] = '0'
        tensorsharp.verify_profile(argv, env, planned)

    def test_reference_no_spec_rejects_cli_and_inherited_activation(self):
        planned = self.profile()
        for tail in (['--spec-type', 'ngram-simple'], ['--spec-default'],
                     ['--spec_type', 'ngram-simple'], ['--model-draft', '/draft.gguf'],
                     ['-md', '/draft.gguf'], ['--spec-type', 'none', '--spec-type', 'ngram-mod']):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                observer.verify_arguments(self.argv() + tail, planned)
        for key, value in (('LLAMA_ARG_SPEC_TYPE', 'ngram-simple'),
                           ('LLAMA_ARG_SPEC_DRAFT_MODEL', '/draft.gguf'),
                           ('LLAMA_ARG_DRAFT_MAX', '8')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                observer.verify_no_speculation(self.argv() + ['--spec-type', 'none'], planned, {key:value})
        observer.verify_no_speculation(self.argv(), planned, {})
        observer.verify_no_speculation(self.argv() + ['--spec-type', 'none'], planned,
                                       {'LLAMA_ARG_SPEC_TYPE':'none'})

    def eight_device_profile(self, tensor=False):
        planned, argv, env = self.ts_profile()
        ranges = [[0, 19]] + [[20 + i * 6, 25 + i * 6] for i in range(7)]
        planned['settings'].pop('layer_ranges')
        planned['settings'].update(gpu_count=8, visible_gpu_devices=[0,1,2,3,4,5,6,8],
                                   cpu_threads=48, cpu_expert_layer_count=16)
        distribution = {'mode': 'tensor' if tensor else 'layer', 'degree': 8}
        if not tensor:distribution['layer_ranges'] = ranges
        planned['engine_options']['distribution'] = distribution
        argv[argv.index('--layer-split') + 1] = '8'
        if tensor:argv[argv.index('--layer-split')] = '--tp'
        argv[argv.index('--cpu-moe-threads') + 1] = '48'
        argv[argv.index('--n-cpu-moe') + 1] = '16'
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,8', TS_CPU_MOE_THREADS='48',
                   TS_GGML_CPU_THREADS='48', OMP_NUM_THREADS='48')
        if tensor:env.pop('TS_MUSE_SPARK_LAYER_SPLIT')
        else:env['TS_MUSE_SPARK_LAYER_SPLIT'] = '20,6,6,6,6,6,6,6'
        return planned, argv, env

    def test_eight_rank_tensor_and_layer_profiles_keep_common_work_equal(self):
        layer, layer_argv, layer_env = self.eight_device_profile()
        tensor, tensor_argv, tensor_env = self.eight_device_profile(tensor=True)
        self.assertEqual(layer['settings'], tensor['settings'])
        checked_layer, _ = tensorsharp.verify_profile(layer_argv, layer_env, layer)
        checked_tensor, _ = tensorsharp.verify_profile(tensor_argv, tensor_env, tensor)
        self.assertEqual(8, checked_layer['--layer-split'])
        self.assertEqual(8, checked_tensor['--tp'])
        self.assertNotIn('layer_ranges', tensor['engine_options']['distribution'])

    def test_nine_rank_profile_allows_uneven_partitions_without_inventing_ranges(self):
        planned, argv, env = self.eight_device_profile(tensor=True)
        planned['settings'].update(gpu_count=9, visible_gpu_devices=list(range(9)))
        planned['engine_options']['distribution']['degree'] = 9
        argv[argv.index('--tp') + 1] = '9'
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(9)))
        tensorsharp.verify_profile(argv, env, planned)

    def test_wrong_physical_devices_degree_or_conflicting_modes_fail(self):
        for mutate in (
            lambda p,a,e: e.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),
            lambda p,a,e: p['engine_options']['distribution'].update(degree=9),
            lambda p,a,e: a.extend(['--layer-split','8']),
            lambda p,a,e: e.update(TS_MUSE_SPARK_LAYER_SPLIT='20,6,6,6,6,6,6,6'),
            lambda p,a,e: e.update(TENSORSHARP_LAYER_SPLIT_DEGREE='8'),
            lambda p,a,e: p['settings'].update(visible_gpu_devices=[0]*8),
        ):
            with self.subTest(mutate=mutate):
                planned, argv, env = self.eight_device_profile(tensor=True)
                mutate(planned, argv, env)
                with self.assertRaises(ValueError):tensorsharp.verify_profile(argv, env, planned)

    def test_partition_gaps_overlap_or_incorrect_cover_fail(self):
        for ranges in ([[0,19],[19,25]] + [[26+i*6,31+i*6] for i in range(6)],
                       [[0,19]] + [[20+i*6,25+i*6] for i in range(6)],
                       [[0,20]] + [[21+i*6,26+i*6] for i in range(7)]):
            planned, argv, env = self.eight_device_profile()
            planned['engine_options']['distribution']['layer_ranges'] = ranges
            with self.assertRaises(ValueError):tensorsharp.verify_profile(argv, env, planned)

    def test_eight_rank_reference_still_counts_output_and_rejects_unobserved_tensor_mode(self):
        planned, _, _ = self.eight_device_profile()
        planned['server_url'] = 'http://127.0.0.1:8082'
        planned['settings']['prefix_cache'] = True
        argv = self.argv()
        argv += ['--spec-type', 'none']
        for flag, value in (('--n-cpu-moe','16'),('--threads','48'),('--threads-batch','48'),
                            ('--tensor-split','20,6,6,6,6,6,6,7'),
                            ('--device',','.join(f'CUDA{i}' for i in range(8)))):
            argv[argv.index(flag)+1] = value
        checked = observer.verify_arguments(argv, planned)
        self.assertEqual('20,6,6,6,6,6,6,7', checked['--tensor-split'])
        planned['engine_options']['distribution'] = {'mode':'tensor','degree':8}
        with self.assertRaises(ValueError):observer.verify_arguments(argv, planned)

    def test_new_reference_profile_requires_explicit_no_speculation(self):
        planned = self.profile()
        planned['engine_options'] = {'distribution': {
            'mode':'layer', 'degree':5, 'layer_ranges':planned['settings']['layer_ranges']}}
        with self.assertRaises(ValueError):
            observer.verify_arguments(self.argv(), planned)
        observer.verify_arguments(self.argv() + ['--spec-type', 'none'], planned)

    def test_binary_changes_are_rejected_without_replacing_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'llama-server';path.write_bytes(b'verified')
            report={'binary_hashes':{str(path):hashlib.sha256(b'verified').hexdigest()}}
            self.assertEqual(report['binary_hashes'],observer.verify_binaries(report))
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):observer.verify_binaries(report)

    def test_readiness_port_must_belong_to_observed_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);process=root/'123';(process/'fd').mkdir(parents=True)
            (process/'net').mkdir();(process/'fd'/'5').touch()
            # /proc/net/tcp covers the whole namespace, including unrelated PIDs.
            (process/'net'/'tcp').write_text('header\n0: 0100007F:1F92 00000000:0000 0A 0:0 0:0 0 0 0 456\n')
            with patch.object(observer.os,'readlink',return_value='socket:[999]'):
                self.assertFalse(observer.owns_listen_port(123,8082,root))
            with patch.object(observer.os,'readlink',return_value='socket:[456]'):
                self.assertTrue(observer.owns_listen_port(123,8082,root))
                self.assertFalse(observer.owns_listen_port(123,8083,root))


if __name__=='__main__':
    unittest.main()
