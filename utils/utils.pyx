import numpy as np
cimport numpy as np
cimport cython
ctypedef fused numeric:
    np.int8_t
    np.complex64_t

def flip_wrapped(source, pad_size, pos):
    order = len(source.shape)-1
    if order == 1:
        return flip_wrapped_1d(source, pad_size, pos)
    elif order == 2:
        return flip_wrapped_2d(source, pad_size, pos)
    elif order == 3:
        return flip_wrapped_3d(source, pad_size, pos)
    else:
        raise ValueError("Order 1+%d not supported" % order)

def replace_wrapped(old, replacement, center):
    assert len(old.shape) == len(replacement.shape), 'old and replacement should have same order'
    order = len(old.shape)-1
    if order == 1:
        return replace_wrapped_1d(old, replacement, center)
    elif order == 2:
        return replace_wrapped_2d(old, replacement, center)
    elif order == 3:
        return replace_wrapped_3d(old, replacement, center)
    else:
        raise ValueError("Order 1+%d not supported" % order)

def window(source, size, center):
    order = len(source.shape)-1
    if order == 1:
        return window_1d(source, size, center)
    elif order == 2:
        return window_2d(source, size, center)
    elif order == 3:
        return window_3d(source, size, center)
    else:
        raise ValueError("Order 1+%d not supported" % order)

def all_windows(source, window_size):
    order = len(source.shape)-1
    if order == 1:
        return all_windows_1d(source, window_size)
    elif order == 2:
        return all_windows_2d(source, window_size)
    elif order == 3:
        return all_windows_3d(source, window_size)
    else:
        raise ValueError("Order 1+%d not supported" % order)

# @cython.boundscheck(False)
@cython.wraparound(False)
def flip_wrapped_1d(np.ndarray[np.int8_t, ndim=2] source,
                     np.ndarray[np.int_t, ndim=1] pad_size,
                     np.ndarray[np.int_t, ndim=2] pos):
    assert source.shape[0] == pos.shape[0], 'batch size of source and pos do not match'
    assert pad_size.shape[0] == 1,          'pad size should be of shape (N, 1)'
    assert pos.shape[1] == 1,               'pos should be of shape (N, 1)'
    cdef int wrap_x
    cdef int x
    cdef int n, N = source.shape[0]
    cdef int size_x = source.shape[1]-2*pad_size[0]
    for n in range(N):
        for wrap_x in range(-1, 2):
            x = pad_size[0]+wrap_x*size_x+pos[n, 0]
            if 0 <= x and x < source.shape[1]:
                source[n, x] *= -1

# # @cython.boundscheck(False)
@cython.wraparound(False)
def flip_wrapped_2d(np.ndarray[np.int8_t, ndim=3] source,
                     np.ndarray[np.int_t, ndim=1] pad_size,
                     np.ndarray[np.int_t, ndim=2] pos):
    assert source.shape[0] == pos.shape[0], 'batch size of source and pos do not match'
    assert pad_size.shape[0] == 2,          'pad size should be of shape (N, 2)'
    assert pos.shape[1] == 2,               'pos should be of shape (N, 2)'
    cdef int wrap_x, wrap_y
    cdef int x, y
    cdef int n, N = source.shape[0]
    cdef int size_x = source.shape[2]-2*pad_size[1]
    cdef int size_y = source.shape[1]-2*pad_size[0]
    for n in range(N):
        for wrap_y in range(-1, 2):
            y = pad_size[0]+wrap_y*size_y+pos[n, 0]
            if 0 <= y and y < source.shape[1]:
                for wrap_x in range(-1, 2):
                    x = pad_size[1]+wrap_x*size_x+pos[n, 1]
                    if 0 <= x and x < source.shape[2]:
                        source[n, y, x] *= -1

# @cython.boundscheck(False)
@cython.wraparound(False)
def flip_wrapped_3d(np.ndarray[np.int8_t, ndim=4] source,
                     np.ndarray[np.int_t, ndim=1] pad_size,
                     np.ndarray[np.int_t, ndim=2] pos):
    assert source.shape[0] == pos.shape[0], 'batch size of source and pos do not match'
    assert pad_size.shape[0] == 3,          'pad size should be of shape (N, 3)'
    assert pos.shape[1] == 3,               'pos should be of shape (N, 3)'
    cdef int wrap_x, wrap_y, wrap_z
    cdef int x, y, z
    cdef int n, N = source.shape[0]
    cdef int size_x = source.shape[3]-2*pad_size[2]
    cdef int size_y = source.shape[2]-2*pad_size[1]
    cdef int size_z = source.shape[1]-2*pad_size[0]
    for n in range(N):
        for wrap_z in range(-1, 2):
            z = pad_size[0]+wrap_z*size_z+pos[n, 0]
            if 0 <= z and z < source.shape[1]:
                for wrap_y in range(-1, 2):
                    y = pad_size[1]+wrap_y*size_y+pos[n, 1]
                    if 0 <= y and y < source.shape[2]:
                        for wrap_x in range(-1, 2):
                            x = pad_size[2]+wrap_x*size_x+pos[n, 2]
                            if 0 <= x and x < source.shape[3]:
                                source[n, z, y, x] *= -1


# @cython.boundscheck(False)
# @cython.boundcheck(False)
@cython.wraparound(False)
def replace_wrapped_1d(np.ndarray[numeric, ndim=2] old,
                        np.ndarray[numeric, ndim=2] replacement,
                        np.ndarray[np.int_t, ndim=2] center):
    assert old.shape[0] == replacement.shape[0] == center.shape[0], 'old, replacement and center should have same batch size'
    assert center.shape[1] == 1,          'center should be of shape (N,1)'
    assert replacement.shape[1] % 2 == 1, 'dimension 1 of replacement should be odd'
    cdef int wrap_x
    cdef int x
    cdef int n, N = old.shape[0]
    cdef int R_x = replacement.shape[1]
    cdef int pad_x = (R_x-1)//2
    cdef int size_x = old.shape[1]-2*pad_x
    cdef int offset_x
    for n in range(N):
        for wrap_x in range(-1, 2):
            offset_x = int(wrap_x*size_x+center[n, 0])
            for x in range(max(0, -offset_x), min(R_x, old.shape[1]-offset_x)):
                old[n, x+offset_x] = replacement[n, x]


# @cython.boundscheck(False)
@cython.wraparound(False)
def replace_wrapped_2d(np.ndarray[numeric, ndim=3] old,
                        np.ndarray[numeric, ndim=3] replacement,
                        np.ndarray[np.int_t, ndim=2] center):
    assert old.shape[0] == replacement.shape[0] == center.shape[0], 'old, replacement and center should have same batch size'
    assert center.shape[1] == 2,          'center should be of shape (N,2)'
    assert replacement.shape[1] % 2 == 1, 'dimension 1 of replacement should be odd'
    assert replacement.shape[2] % 2 == 1, 'dimension 2 of replacement should be odd'
    cdef int wrap_x, wrap_y
    cdef int x, y
    cdef int n, N = old.shape[0]
    cdef int R_x = replacement.shape[2]
    cdef int R_y = replacement.shape[1]
    cdef int pad_x = (R_x-1)//2
    cdef int pad_y = (R_y-1)//2
    cdef int size_x = old.shape[2]-2*pad_x
    cdef int size_y = old.shape[1]-2*pad_y
    cdef int offset_x, offset_y
    for n in range(N):
        for wrap_y in range(-1, 2):
            offset_y = int(wrap_y*size_y+center[n, 0])
            for y in range(max(0, -offset_y), min(R_y, old.shape[1]-offset_y)):
                for wrap_x in range(-1, 2):
                    offset_x = int(wrap_x*size_x+center[n, 1])
                    for x in range(max(0, -offset_x), min(R_x, old.shape[2]-offset_x)):
                        old[n, y+offset_y, x+offset_x] = replacement[n, y, x]



# @cython.boundscheck(False)
@cython.wraparound(False)
def replace_wrapped_3d(np.ndarray[numeric, ndim=4] old,
                        np.ndarray[numeric, ndim=4] replacement,
                        np.ndarray[np.int_t, ndim=2] center):
    assert old.shape[0] == replacement.shape[0] == center.shape[0], 'old, replacement and center should have same batch size'
    assert center.shape[1] == 3,          'center should be of shape (N,3)'
    assert replacement.shape[1] % 2 == 1, 'dimension 1 of replacement should be odd'
    assert replacement.shape[2] % 2 == 1, 'dimension 2 of replacement should be odd'
    assert replacement.shape[3] % 2 == 1, 'dimension 3 of replacement should be odd'
    cdef int wrap_x, wrap_y, wrap_z
    cdef int x, y, z
    cdef int n, N = old.shape[0]
    cdef int R_x = replacement.shape[3]
    cdef int R_y = replacement.shape[2]
    cdef int R_z = replacement.shape[1]
    cdef int pad_x = (R_x-1)//2
    cdef int pad_y = (R_y-1)//2
    cdef int pad_z = (R_z-1)//2
    cdef int size_x = old.shape[3]-2*pad_x
    cdef int size_y = old.shape[2]-2*pad_y
    cdef int size_z = old.shape[1]-2*pad_z
    cdef int offset_x, offset_y, offset_z
    for n in range(N):
        for wrap_z in range(-1, 2):
            offset_z = int(wrap_z*size_z+center[n, 0])
            for z in range(max(0, -offset_z), min(R_z, old.shape[1]-offset_z)):
                for wrap_y in range(-1, 2):
                    offset_y = int(wrap_y*size_y+center[n, 1])
                    for y in range(max(0, -offset_y), min(R_y, old.shape[2]-offset_y)):
                        for wrap_x in range(-1, 2):
                            offset_x = int(wrap_x*size_x+center[n, 2])
                            for x in range(max(0, -offset_x), min(R_x, old.shape[3]-offset_x)):
                                old[n, z+offset_z, y+offset_y, x+offset_x] = replacement[n, z, y, x]


# @cython.boundscheck(False)
@cython.wraparound(False)
def window_1d(np.ndarray[numeric, ndim=2] source,
              np.ndarray[np.int_t, ndim=1] size,
              np.ndarray[np.int_t, ndim=2] center):
    assert source.shape[0] == center.shape[0], 'source and center should have same batch size'
    assert size.shape[0] == 1,                 'size should be of shape (1)'
    assert center.shape[1] == 1,               'center should be of shape (N,1)'
    assert size[0] % 2 == 1,                   'size[0] should be odd'

    cdef int n, N = source.shape[0]
    cdef int x
    cdef int offset_x

    cdef np.ndarray[numeric, ndim=2] out = np.zeros([N, size[0]], dtype=source.dtype)

    for n in range(N):
        offset_x = center[n, 0]
        for x in range(size[0]):
            out[n, x] = source[n, x+offset_x]
    return out


# @cython.boundscheck(False)
@cython.wraparound(False)
def window_2d(np.ndarray[numeric, ndim=3] source,
              np.ndarray[np.int_t, ndim=1] size,
              np.ndarray[np.int_t, ndim=2] center):
    assert source.shape[0] == center.shape[0], 'source and center should have same batch size'
    assert size.shape[0] == 2,                 'size should be of shape (2)'
    assert center.shape[1] == 2,               'center should be of shape (N,2)'
    assert size[0] % 2 == 1,                   'size[0] should be odd'
    assert size[1] % 2 == 1,                   'size[1] should be odd'

    cdef int n, N = source.shape[0]
    cdef int x, y
    cdef int offset_x, offset_y

    cdef np.ndarray[numeric, ndim=3] out = np.zeros([N, size[0], size[1]], dtype=source.dtype)

    for n in range(N):
        offset_y = center[n, 0]
        for y in range(size[0]):
            offset_x = center[n, 1]
            for x in range(size[1]):
                out[n, y, x] = source[n, y+offset_y, x+offset_x]
    return out


# @cython.boundscheck(False)
@cython.wraparound(False)
def window_3d(np.ndarray[numeric, ndim=4] source,
              np.ndarray[np.int_t, ndim=1] size,
              np.ndarray[np.int_t, ndim=2] center):
    assert source.shape[0] == center.shape[0], 'source and center should have same batch size'
    assert size.shape[0] == 3,                 'size should be of shape (3)'
    assert center.shape[1] == 3,               'center should be of shape (N,3)'
    assert size[0] % 2 == 1,                   'size[0] should be odd'
    assert size[1] % 2 == 1,                   'size[1] should be odd'
    assert size[2] % 2 == 1,                   'size[2] should be odd'

    cdef int n, N = source.shape[0]
    cdef int x, y, z
    cdef int offset_x, offset_y, offset_z

    cdef np.ndarray[numeric, ndim=4] out = np.zeros([N, size[0], size[1], size[2]], dtype=source.dtype)

    for n in range(N):
        offset_z = center[n, 0]
        for z in range(size[0]):
            offset_y = center[n, 1]
            for y in range(size[1]):
                offset_x = center[n, 2]
                for x in range(size[2]):
                    out[n, z, y, x] = source[n, z+offset_z, y+offset_y, x+offset_x]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def all_windows_1d(np.ndarray[numeric, ndim=2] source,
                   np.ndarray[np.int_t, ndim=1] window_size):
    assert window_size.shape[0] == 1, 'window_size should be of shape (1)'
    assert window_size[0] % 2 == 1,   'window_size[0] should be odd'

    cdef int n, N = source.shape[0]
    cdef int i
    cdef int x
    cdef int pad_x = (window_size[0]-1)//2
    cdef int size_x = source.shape[1] - 2 * pad_x # Unpadded source size

    cdef np.ndarray[numeric, ndim=3] out = np.zeros([N, size_x, window_size[0]], dtype=source.dtype)

    for n in range(N):
        for i in range(size_x):
            for x in range(window_size[0]):
                out[n, i, x] = source[n, i+x]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def all_windows_2d(np.ndarray[numeric, ndim=3] source,
                   np.ndarray[np.int_t, ndim=1] window_size):
    assert window_size.shape[0] == 2, 'window_size should be of shape (2)'
    assert window_size[0] % 2 == 1,   'window_size[0] should be odd'
    assert window_size[1] % 2 == 1,   'window_size[1] should be odd'

    cdef int n, N = source.shape[0]
    cdef int i, j
    cdef int x, y
    cdef int pad_x = (window_size[1]-1)//2
    cdef int size_x = source.shape[2] - 2 * pad_x # Unpadded source size
    cdef int pad_y = (window_size[0]-1)//2
    cdef int size_y = source.shape[1] - 2 * pad_y # Unpadded source size

    cdef np.ndarray[numeric, ndim=5] out = np.zeros(
        [N, size_y, size_x, window_size[0], window_size[1]], dtype=source.dtype)

    for n in range(N):
        for j in range(size_y):
            for y in range(window_size[0]):
                for i in range(size_x):
                    for x in range(window_size[1]):
                        out[n, j, i, y, x] = source[n, j+y, i+x]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def all_windows_3d(np.ndarray[numeric, ndim=4] source,
                   np.ndarray[np.int_t, ndim=1] window_size):
    assert window_size.shape[0] == 3, 'window_size should be of shape (3)'
    assert window_size[0] % 2 == 1,   'window_size[0] should be odd'
    assert window_size[1] % 2 == 1,   'window_size[1] should be odd'
    assert window_size[2] % 2 == 1,   'window_size[2] should be odd'

    cdef int n, N = source.shape[0]
    cdef int i, j, l
    cdef int x, y, z
    cdef int pad_x = (window_size[2]-1)//2
    cdef int size_x = source.shape[3] - 2 * pad_x # Unpadded source size
    cdef int pad_y = (window_size[1]-1)//2
    cdef int size_y = source.shape[2] - 2 * pad_y # Unpadded source size
    cdef int pad_z = (window_size[0]-1)//2
    cdef int size_z = source.shape[1] - 2 * pad_z # Unpadded source size

    cdef np.ndarray[numeric, ndim=7] out = np.zeros(
        [N, size_y, size_x, size_y, window_size[0], window_size[1], window_size[2]],
        dtype=source.dtype)

    for n in range(N):
        for l in range(size_z):
            for z in range(window_size[0]):
                for j in range(size_y):
                    for y in range(window_size[1]):
                        for i in range(size_x):
                            for x in range(window_size[2]):
                                out[n, l, j, i, z, y, x] = source[n, l+z, j+y, i+x]
    return out
