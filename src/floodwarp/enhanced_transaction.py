"""Bit-exact batched device copies for transactional component trials."""
from typing import Any
import warp as wp


@wp.kernel
def copy_fields(a0: wp.array(dtype=Any), a1: wp.array(dtype=Any), a2: wp.array(dtype=Any), a3: wp.array(dtype=Any),
                a4: wp.array(dtype=Any), a5: wp.array(dtype=Any), a6: wp.array(dtype=Any), a7: wp.array(dtype=Any),
                b0: wp.array(dtype=Any), b1: wp.array(dtype=Any), b2: wp.array(dtype=Any), b3: wp.array(dtype=Any),
                b4: wp.array(dtype=Any), b5: wp.array(dtype=Any), b6: wp.array(dtype=Any), b7: wp.array(dtype=Any),
                restore: int):
    i=wp.tid()
    if i<a0.shape[0]:
        if restore==1:a0[i]=b0[i]
        else:b0[i]=a0[i]
    if i<a1.shape[0]:
        if restore==1:a1[i]=b1[i]
        else:b1[i]=a1[i]
    if i<a2.shape[0]:
        if restore==1:a2[i]=b2[i]
        else:b2[i]=a2[i]
    if i<a3.shape[0]:
        if restore==1:a3[i]=b3[i]
        else:b3[i]=a3[i]
    if i<a4.shape[0]:
        if restore==1:a4[i]=b4[i]
        else:b4[i]=a4[i]
    if i<a5.shape[0]:
        if restore==1:a5[i]=b5[i]
        else:b5[i]=a5[i]
    if i<a6.shape[0]:
        if restore==1:a6[i]=b6[i]
        else:b6[i]=a6[i]
    if i<a7.shape[0]:
        if restore==1:a7[i]=b7[i]
        else:b7[i]=a7[i]


def prepare(*, solver, saved):
    groups=[]
    names=sorted(saved,key=saved_field_sort_key)
    for start in range(0,len(names),8):
        selected=names[start:start+8]
        # Duplicate the final pair to fill the internal eight-field ABI.
        # Copies are idempotent; no extra scratch or unsafe pointer views.
        selected.extend([selected[-1]]*(8-len(selected)))
        current=[getattr(solver,name).flatten() for name in selected]
        snapshots=[saved[name].flatten() for name in selected]
        inputs=current+snapshots
        signature={}
        for i,field in enumerate(current):
            signature['a'+str(i)]=wp.array(dtype=field.dtype)
            signature['b'+str(i)]=wp.array(dtype=field.dtype)
        wp.overload(copy_fields,signature)
        groups.append((max(field.size for field in current),inputs))
    wp.load_module(module=copy_fields.module,device=solver.device)
    return groups


def saved_field_sort_key(name):
    return name


def copy(*, groups, restore, device):
    for size,fields in groups:
        wp.launch(kernel=copy_fields,dim=size,inputs=[*fields,int(restore)],device=device)
