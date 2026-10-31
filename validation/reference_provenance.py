"""Read-only verification of retained FV1 operators and boundary adapters."""
from pathlib import Path
import difflib
import hashlib
import subprocess

from validation.benchmark_restart import publish_json


def record_reference(*,upstream,source_prefix,adapted,report_path,diff_path,paths):
    revision=subprocess.run(args=['git','--no-optional-locks','-C',str(upstream),'rev-parse','HEAD'],
        capture_output=True,text=True,check=True).stdout.strip()
    matches={};patch=[]
    for name in paths:
        original=subprocess.run(args=['git','--no-optional-locks','-C',str(upstream),
            'show',revision+':'+source_prefix+name],capture_output=True,text=True,check=True).stdout
        current=(adapted/name).read_text()
        matches[name]=dict(normalised_text_matches=original==current,
            upstream_normalised_sha256=hashlib.sha256(original.encode()).hexdigest(),
            adapted_normalised_sha256=hashlib.sha256(current.encode()).hexdigest())
        patch.extend(difflib.unified_diff(original.splitlines(keepends=True),current.splitlines(keepends=True),
            fromfile='upstream/'+name,tofile='retained-adapter/'+name))
    diff_path.write_text(''.join(patch))
    publish_json(path=report_path,value=dict(retained_upstream_revision=revision,
        upstream_checkout=str(upstream),adapted_source=str(adapted),files=matches,
        scope='File-level verification against retained upstream source; explicit boundary and diagnostic changes are preserved in the accompanying diff.'))


if __name__ == "__main__":
    upstream=Path('/home/patrickdurney/PycharmProjects/Overlander/.external/purau_lisflood_fp64_20261023/upstream-8.3')
    source_prefix='source/'
    adapted=Path('outputs/qualified_reference/source')
    report_path=Path('validation/repair_review/reference_revision.json')
    diff_path=Path('validation/repair_review/reference_operator_adapter.patch')
    paths=('cuda/cuda_hll.cu','cuda/cuda_hll.cuh','cuda/fv1/cuda_fv1_solver.cu','cuda/cuda_solver.cu')
    record_reference(upstream=upstream,source_prefix=source_prefix,adapted=adapted,
        report_path=report_path,diff_path=diff_path,paths=paths)
