import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('hidden_transfers',Path(__file__).with_name('embeddinggemma2-cupti-hidden-transfers.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HiddenTransferTests(unittest.TestCase):
    def fixture(self):
        def event(start,end,**extra):
            return dict(start_ns=start,end_ns=end,device=0,context=1,stream=13,**extra)
        def copy_event(start,direction):
            return event(start,start+1,direction=direction,bytes=36*768*4,copy_count=1)
        copies=[copy_event(1,'HtoD'),copy_event(3,'DtoH')]
        kernels=[]
        for i in range(16):
            start=10+i*10
            copies.extend([copy_event(start,'HtoD'),copy_event(start+5,'DtoH')])
            kernels.append(event(start+2,start+3,name='vision_kernel<64>'))
        return dict(schema_version=2,completed=True,diagnostic=True,hardware_counters=False,cupti_api_version=130401,errors=[],dropped=0,incomplete=0,callback_exceptions=0,truncated=0,invalid_buffers=0,kernel_count=16,memcpy_record_count=34,kernel_events=kernels,memcpy_events=copies)

    def test_patch_pair_is_excluded_and_all_blocks_are_bracketed(self):
        result=module.attribute(self.fixture(),36)
        self.assertEqual(result['total_block_copy_bytes'],2*16*36*768*4)
        self.assertEqual(result['block_transfers']['HtoD']['count'],16)
        self.assertFalse(result['paired_performance_claim'])

    def test_missing_reordered_cross_context_and_batched_records_reject(self):
        for mutation in (lambda d:d['memcpy_events'].pop(),
                         lambda d:d['memcpy_events'][3].update(start_ns=10),
                         lambda d:d['memcpy_events'][3].update(context=2),
                         lambda d:d['memcpy_events'][3].update(copy_count=2),
                         lambda d:d.update(incomplete=1)):
            report=copy.deepcopy(self.fixture());mutation(report)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                module.attribute(report,36)

    def test_cuda12_api26_capture_is_rejected(self):
        report=self.fixture();report['cupti_api_version']=26
        with self.assertRaisesRegex(ValueError,'CUDA 13'):
            module.attribute(report,36)


if __name__=='__main__':
    unittest.main()
