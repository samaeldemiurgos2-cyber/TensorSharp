"""Bounded negative controls; no model, GPU, VM or external HTTP required."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'validation/muse-spark-speculation-regression.py'
SPEC=importlib.util.spec_from_file_location('spark_spec_regression',PATH)
tool=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(tool)


def logs(ngram=False,request='chat-test',accepted=3,drafted=5,rollbacks=1):
    path='SpeculativePerSequence' if ngram else 'BatchedPaged'
    text=f'chat.start arch=muse_spark\n[cb] step#1 {path} scheduled=1 served=1 | {request}:D fwd=3 computed=20 in=42\n'
    if ngram:
        text+=(f'Speculative decoding armed for {request} (algorithm=ngram(n=2..8), maxDraft=8)\n'
               f'Speculative decoding stats for {request}: drafted={drafted} accepted={accepted} acceptance=60% '
               f'verifySteps=2 plainSteps=1 rollbacks={rollbacks}\n')
    return text


class SpeculationRegressionTests(unittest.TestCase):
    def test_correlated_rejection_is_required(self):
        self.assertFalse(tool.telemetry_failures(tool.telemetry(logs(True)),1,'ngram_reject'))
        for text in (logs(True).replace('stats for chat-test','stats for other'),
                     logs(True).replace('armed for chat-test','armed for other'),
                     logs(True)+logs(True,request='foreign'),logs(True).replace('chat.start arch=','missing arch='),
                     logs(True).replace('SpeculativePerSequence','PerSequence')):
            self.assertTrue(tool.telemetry_failures(tool.telemetry(text),1,'ngram_reject'))

    def test_unexercised_is_distinct_from_content_failure(self):
        failures=tool.telemetry_failures(tool.telemetry(logs(True,accepted=5,drafted=5,rollbacks=0)),1,'ngram_reject')
        self.assertTrue(failures)
        self.assertTrue(all(x.startswith('not_exercised:') for x in failures))

    def test_prefill_rearming_same_request_is_allowed_but_foreign_id_is_not(self):
        repeated=logs(True)+'Speculative decoding armed for chat-test (algorithm=ngram(n=2..8), maxDraft=8)\n'
        self.assertFalse(tool.telemetry_failures(tool.telemetry(repeated),1,'ngram_reject'))
        foreign=repeated+'Speculative decoding armed for foreign (algorithm=ngram(n=2..8), maxDraft=8)\n'
        self.assertTrue(tool.telemetry_failures(tool.telemetry(foreign),1,'ngram_reject'))

    def test_impossible_stats_and_failed_steps_rejected(self):
        for text in (logs(True,accepted=6),logs(True,rollbacks=3),logs(True).replace('fwd=3','fwd=0'),
                     logs(True).replace('computed=20','error=NativeError computed=20'),
                     logs(True).replace('served=1','served=2')):
            self.assertTrue(tool.telemetry_failures(tool.telemetry(text),1,'ngram_accept'))

    def test_mixed_step_requires_distinct_members(self):
        prefix='chat.start arch=muse_spark\nchat.start arch=muse_spark\n'
        valid=prefix+'[cb] step#4 BatchedPaged scheduled=2 served=2 | a:P fwd=128 | b:D fwd=1\n'
        self.assertFalse(tool.telemetry_failures(tool.telemetry(valid),2,'mixed'))
        for invalid in (valid.replace('b:D','a:D'),valid.replace('b:D','b:P'),valid.replace('scheduled=2','scheduled=1'),
                        valid.replace(' | b:D fwd=1',''),valid.replace('BatchedPaged','PerSequence')):
            self.assertTrue(tool.telemetry_failures(tool.telemetry(invalid),2,'mixed'))

    def test_staggered_shared_harness_warmup_is_counted_as_a_fifth_request(self):
        text=logs(False,request='warmup')
        text+='chat.start arch=muse_spark\n'*4
        text+='[cb] step#2 BatchedPaged scheduled=4 served=4 | long:D fwd=1 | s1:P fwd=20 | s2:P fwd=20 | s3:P fwd=20\n'
        observed=tool.telemetry(text)
        self.assertFalse(tool.telemetry_failures(observed,5,'mixed'))
        self.assertTrue(tool.telemetry_failures(observed,4,'mixed'))

    def metrics(self,case):
        return dict(assistant_message={'content':case['expected'][:17] if case['name']=='limit17' else case['expected']},
                    usage_present=True,finish_reason='length' if case['name']=='limit17' else 'stop',
                    prompt_tokens=200,completion_tokens=17 if case['name']=='limit17' else 100,
                    raw_usage={'prompt_tokens_details':{'cached_tokens':0}})

    def test_content_and_limit_are_independently_checked(self):
        wire={'sse_lines':['data: [DONE]']}
        for case in tool.fixtures('spark'):
            metrics=self.metrics(case)
            self.assertFalse(tool.content_failures(case,metrics,wire))
            for altered in (metrics|{'completion_tokens':0},metrics|{'finish_reason':'error'},
                            metrics|{'assistant_message':{'content':'wrong'}},
                            metrics|{'raw_usage':{'prompt_tokens_details':{'cached_tokens':False}}}):
                self.assertTrue(tool.content_failures(case,altered,wire))
            self.assertTrue(tool.content_failures(case,metrics,{}))

    def test_idle_checks_and_extra_completions(self):
        idle=dict(busy=False,processing=0,pending_requests=0,total_processed=1)
        tool.require_idle(idle)
        for status in (idle|{'processing':1},idle|{'pending_requests':1},idle|{'busy':True}):
            with self.assertRaises(ValueError): tool.require_idle(status)
        with patch.object(tool,'queue_status',return_value=idle|{'total_processed':2}):
            with self.assertRaisesRegex(ValueError,'Unexpected concurrent'): tool.wait_idle('unused',1)

    def test_log_rotation_or_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'log'; path.write_text('old telemetry\n')
            cursor=tool.fresh_cursor(path)
            with path.open('a') as output: output.write(logs(True))
            self.assertEqual(logs(True),tool.fresh_log(path,cursor).replace('\r\n','\n'))
            path.write_text('')
            with self.assertRaises(ValueError): tool.fresh_log(path,cursor)

    def setup_run(self,root,phase,reference=None):
        log=root/'server.log'; log.touch(exist_ok=True)
        provenance=dict(launch_observed=True,server_url='http://unused',server_log=str(log),
            binary_hashes={'host':'sha'},weights_sha256={'model':'sha'},
            settings=dict(key_cache_dtype='f32',value_cache_dtype='f32',prefix_cache=False,
                          speculation='disabled' if phase=='nospec' else 'ngram'),
            engine_options={'environment':{'TS_SPEC':'0' if phase=='nospec' else '1','TS_MUSE_SPARK_FUSE_RESIDUALS':'1'}})
        prov=root/(phase+'-observed.json'); prov.write_text(json.dumps(provenance))
        return types.SimpleNamespace(phase=phase,url='http://unused',model='spark',server_log=str(log),
            provenance=str(prov),out=str(root/(phase+'.json')),reference=str(reference) if reference else None,timeout=1)

    def invoke(self,args,*,reject=True,wrong=False,stale=False):
        fixtures=tool.fixtures(args.model); sent=[]
        def completion(url,body,timeout,wire):
            case=fixtures[len(sent)]; sent.append(body)
            self.assertEqual(case['request'],body)
            wire['sse_lines']=['data: [DONE]']
            if not stale:
                with Path(args.server_log).open('a') as output:
                    output.write(logs(args.phase=='ngram',request='chat-'+case['name'],
                                      accepted=3 if reject else 5,rollbacks=1 if reject else 0))
            metrics=self.metrics(case)
            if wrong: metrics['assistant_message']['content']='wrong'
            return metrics
        with patch.object(tool,'live_identity',return_value={'pid':1,'start_ticks':2}), \
             patch.object(tool,'wait_idle',side_effect=lambda url,expected_total=None:dict(
                 busy=False,processing=0,pending_requests=0,total_processed=len(sent))), \
             patch.object(tool.engines,'_run_streaming',side_effect=completion), \
             patch.object(tool.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
            code=tool.run(args)
        return code,json.loads(Path(args.out).read_text()),sent

    def test_both_phases_complete_and_request_bodies_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); plain=self.setup_run(root,'nospec')
            self.assertEqual(0,self.invoke(plain)[0])
            ngram=self.setup_run(root,'ngram',plain.out)
            code,result,sent=self.invoke(ngram)
            self.assertEqual(0,code); self.assertEqual(3,len(sent)); self.assertTrue(result['passed'])
            self.assertEqual(['edited_copy','after_rollback','limit17'],[x['name'] for x in result['cases']])

    def test_correct_output_without_rollback_is_preserved_as_not_exercised(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); plain=self.setup_run(root,'nospec'); self.invoke(plain)
            code,result,sent=self.invoke(self.setup_run(root,'ngram',plain.out),reject=False)
            self.assertEqual(1,code); self.assertEqual(3,len(sent))
            self.assertEqual('not_exercised',result['cases'][0]['status'])

    def test_failed_reference_blocks_all_ngram_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); plain=self.setup_run(root,'nospec'); self.invoke(plain,wrong=True)
            code,result,sent=self.invoke(self.setup_run(root,'ngram',plain.out))
            self.assertEqual(1,code); self.assertEqual([],sent)
            self.assertIn('fully passing',str(result['failures']))

    def test_old_telemetry_does_not_pass_and_results_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.setup_run(Path(directory),'nospec')
            Path(args.server_log).write_text(logs(False))
            code,result,_=self.invoke(args,stale=True)
            self.assertEqual(1,code)
            preserved=Path(args.out).read_bytes()
            with self.assertRaises(FileExistsError): self.invoke(args)
            self.assertEqual(preserved,Path(args.out).read_bytes())

    def test_binary_and_precision_changes_block_reference(self):
        cases=[]
        for case in tool.fixtures('spark'):
            metrics=self.metrics(case)
            cases.append(dict(case,status='passed',output=metrics['assistant_message']['content'],metrics=metrics,
                wire={'sse_lines':['data: [DONE]']},telemetry=tool.telemetry(logs(False))))
        before=dict(passed=True,phase='nospec',cases=cases,provenance=dict(binary_hashes={'a':'1'},
            weights_sha256={'b':'2'},settings={'key_cache_dtype':'f32','speculation':'disabled'}))
        after=dict(cases=[],provenance=copy.deepcopy(before['provenance']))
        after['provenance']['settings']['speculation']='ngram'
        self.assertFalse(tool.reference_failures(before,after,'ngram'))
        after['provenance']['settings']['key_cache_dtype']='f16'
        self.assertTrue(tool.reference_failures(before,after,'ngram'))
        after['provenance']['binary_hashes']['a']='3'
        self.assertTrue(tool.reference_failures(before,after,'ngram'))

    def test_forged_pass_flag_does_not_bypass_reference_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); plain=self.setup_run(root,'nospec'); self.invoke(plain)
            result=json.loads(Path(plain.out).read_text())
            result['cases'][0]['metrics']['assistant_message']['content']='wrong'
            Path(plain.out).write_text(json.dumps(result))
            code,result,sent=self.invoke(self.setup_run(root,'ngram',plain.out))
            self.assertEqual(1,code); self.assertEqual([],sent)
            self.assertIn('independently pass',str(result['failures']))


if __name__=='__main__': unittest.main()
