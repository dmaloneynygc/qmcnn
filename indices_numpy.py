"""Test Module."""
import numpy as np

shape = (4, 4)
window_shape = (3, 3)
n = np.prod(shape)
n_window = np.prod(window_shape)
data = np.arange(n).reshape(shape)


def create_index_matrix(data_shape, window_shape):
    """Return wrapped index matrix, shape (n, n_windows)."""
    n_data = np.prod(data_shape)
    n_window = np.prod(window_shape)
    box = np.indices(window_shape)
    index_matrix = np.zeros((n_data, n_window), dtype=np.int32)
    shifts = np.unravel_index(np.arange(n_data), shape)
    offset = (np.array(window_shape)-1)/2
    for i, shift in enumerate(zip(*shifts)):
        shift = np.array(shift)-offset
        window = (box.T + shift).T
        index_matrix[i] = np.ravel_multi_index(window, data_shape, 'wrap') \
            .flatten()
    return index_matrix


index_matrix = create_index_matrix(shape, window_shape)
data_flattened = data.flatten()
data_flattened[index_matrix[[0, 5, 10]]].reshape((-1,)+window_shape)

centers = [0, 5, 15]
batch_size = len(centers)
datas = data_flattened[None, :] + np.arange(batch_size)[:, None] * 100

# %% All windows:
datas[:, index_matrix].reshape((batch_size, n)+window_shape)

# %% Spinflip
new_datas = datas.copy()
new_datas[np.arange(batch_size), np.array(centers)] *= -1
new_datas.reshape((batch_size,)+shape)

# %% Pick different window in each batch sample
selector = np.arange(batch_size)[:, None], index_matrix[centers]
windows = datas[selector].reshape((batch_size,)+window_shape)
windows

# %% Change window
new_windows = (windows * -1).reshape((batch_size, n_window))
new_datas = datas.copy()
new_datas[selector] = new_windows
new_datas.reshape((batch_size,)+shape)
