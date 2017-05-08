import numpy as np
cimport numpy as np
cimport cython

def flip_wrapped(source, pad_size, pos):
    assert len(source.shape) == len(pad_size.shape)
    order = len(source.shape)-1
    if order == 1:
        flip_wrapped_1d(source, pad_size, pos)
    elif order == 2:
        flip_wrapped_2d(source, pad_size, pos)
    elif order == 3:
        flip_wrapped_3d(source, pad_size, pos)
    else:
        raise ValueError("Order 1+%d not supported" % order)

def replace_wrapped(old, replacement, center):
    assert len(old.shape) == len(replacement.shape)
    order = len(old.shape)-1
    if order == 1:
        replace_wrapped_1d(old, replacement, center)
    elif order == 2:
        replace_wrapped_2d(old, replacement, center)
    elif order == 3:
        replace_wrapped_3d(old, replacement, center)
    else:
        raise ValueError("Order 1+%d not supported" % order)

@cython.boundscheck(False)
@cython.wraparound(False)
def flip_wrapped_1d(np.ndarray[np.int8_t, ndim=2] source,
                     int pad_size, np.ndarray[np.int_t, ndim=2] pos):
    assert source.shape[0] == pos.shape[0]
    assert pos.shape[1] == 1
    cdef int wrap_x
    cdef int x
    cdef int n, N = source.shape[0]
    cdef int size_x = source.shape[1]-2*pad_size
    for n in range(N):
        for wrap_x in range(-1, 2):
            x = pad_size+wrap_x*size_x+pos[n, 0]
            if 0 <= x and x < source.shape[1]:
                source[n, x] *= -1

@cython.boundscheck(False)
@cython.wraparound(False)
def flip_wrapped_2d(np.ndarray[np.int8_t, ndim=3] source,
                     int pad_size, np.ndarray[np.int_t, ndim=2] pos):
    assert source.shape[0] == pos.shape[0]
    assert pos.shape[1] == 2
    cdef int wrap_x, wrap_y
    cdef int x, y
    cdef int n, N = source.shape[0]
    cdef int size_x = source.shape[2]-2*pad_size
    cdef int size_y = source.shape[1]-2*pad_size
    for n in range(N):
        for wrap_y in range(-1, 2):
            y = pad_size+wrap_y*size_y+pos[n, 0]
            if 0 <= y and y < source.shape[1]:
                for wrap_x in range(-1, 2):
                    x = pad_size+wrap_x*size_x+pos[n, 1]
                    if 0 <= x and x < source.shape[2]:
                        source[n, y, x] *= -1

@cython.boundscheck(False)
@cython.wraparound(False)
def flip_wrapped_3d(np.ndarray[np.int8_t, ndim=4] source,
                     int pad_size, np.ndarray[np.int_t, ndim=2] pos):
    assert source.shape[0] == pos.shape[0]
    assert pos.shape[1] == 3
    cdef int wrap_x, wrap_y, wrap_z
    cdef int x, y, z
    cdef int n, N = source.shape[0]
    cdef int size_x = source.shape[3]-2*pad_size
    cdef int size_y = source.shape[2]-2*pad_size
    cdef int size_z = source.shape[1]-2*pad_size
    for n in range(N):
        for wrap_z in range(-1, 2):
            z = pad_size+wrap_z*size_z+pos[n, 0]
            if 0 <= z and z < source.shape[1]:
                for wrap_y in range(-1, 2):
                    y = pad_size+wrap_y*size_y+pos[n, 1]
                    if 0 <= y and y < source.shape[2]:
                        for wrap_x in range(-1, 2):
                            x = pad_size+wrap_x*size_x+pos[n, 2]
                            if 0 <= x and x < source.shape[3]:
                                source[n, z, y, x] *= -1

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_wrapped_1d(np.ndarray[np.complex64_t, ndim=2] old,
                        np.ndarray[np.complex64_t, ndim=2] replacement,
                        np.ndarray[np.int_t, ndim=2] center):
    assert old.shape[0] == replacement.shape[0] == center.shape[0]
    assert center.shape[1] == 1
    assert replacement.shape[1] % 2 == 1
    cdef int wrap_x
    cdef int x
    cdef int n, N = old.shape[0]
    cdef int R_x = replacement.shape[1]
    cdef int pad_x = R_x-1
    cdef int size_x = old.shape[1]-2*pad_x
    cdef int offset_x
    for n in range(N):
        for wrap_x in range(-1, 2):
            offset_x = int(pad_x/2+wrap_x*size_x+center[n, 0])
            for x in range(max(offset_x, 0), min(offset_x+R_x, old.shape[1])):
                old[n, x] = replacement[n, x-offset_x]


@cython.boundscheck(False)
@cython.wraparound(False)
def replace_wrapped_2d(np.ndarray[np.complex64_t, ndim=3] old,
                        np.ndarray[np.complex64_t, ndim=3] replacement,
                        np.ndarray[np.int_t, ndim=2] center):
    assert old.shape[0] == replacement.shape[0] == center.shape[0]
    assert center.shape[1] == 2
    assert replacement.shape[1] % 2 == replacement.shape[2] % 2 == 1
    cdef int wrap_x, wrap_y
    cdef int x, y
    cdef int n, N = old.shape[0]
    cdef int R_x = replacement.shape[2]
    cdef int R_y = replacement.shape[1]
    cdef int pad_x = R_x-1
    cdef int pad_y = R_y-1
    cdef int size_x = old.shape[2]-2*pad_x
    cdef int size_y = old.shape[1]-2*pad_y
    cdef int offset_x, offset_y
    for n in range(N):
        for wrap_y in range(-1, 2):
            offset_y = int(pad_y/2+wrap_y*size_y+center[n, 0])
            for y in range(max(offset_y, 0), min(offset_y+R_y, old.shape[1])):
                for wrap_x in range(-1, 2):
                    offset_x = int(pad_x/2+wrap_x*size_x+center[n, 0])
                    for x in range(max(offset_x, 0), min(offset_x+R_x, old.shape[2])):
                        old[n, y, x] = replacement[n, y-offset_y, x-offset_x]



@cython.boundscheck(False)
@cython.wraparound(False)
def replace_wrapped_3d(np.ndarray[np.complex64_t, ndim=4] old,
                        np.ndarray[np.complex64_t, ndim=4] replacement,
                        np.ndarray[np.int_t, ndim=2] center):
    assert old.shape[0] == replacement.shape[0] == center.shape[0]
    assert center.shape[1] == 3
    assert replacement.shape[1] % 2 == 1
    assert replacement.shape[2] % 2 == 1
    assert replacement.shape[3] % 2 == 1
    cdef int wrap_x, wrap_y, wrap_z
    cdef int x, y, z
    cdef int n, N = old.shape[0]
    cdef int R_x = replacement.shape[3]
    cdef int R_y = replacement.shape[2]
    cdef int R_z = replacement.shape[1]
    cdef int pad_x = R_x-1
    cdef int pad_y = R_y-1
    cdef int pad_z = R_z-1
    cdef int size_x = old.shape[3]-2*pad_x
    cdef int size_y = old.shape[2]-2*pad_y
    cdef int size_z = old.shape[1]-2*pad_z
    cdef int offset_x, offset_y, offset_z
    for n in range(N):
        for wrap_z in range(-1, 2):
            offset_z = int(pad_z/2+wrap_z*size_z+center[n, 0])
            for z in range(max(offset_z, 0), min(offset_z+R_z, old.shape[1])):
                for wrap_y in range(-1, 2):
                    offset_y = int(pad_y/2+wrap_y*size_y+center[n, 1])
                    for y in range(max(offset_y, 0), min(offset_y+R_y, old.shape[2])):
                        for wrap_x in range(-1, 2):
                            offset_x = int(pad_x/2+wrap_x*size_x+center[n, 2])
                            for x in range(max(offset_x, 0), min(offset_x+R_x, old.shape[3])):
                                old[n, z, y, x] = replacement[n, z-offset_z, y-offset_y, x-offset_x]
