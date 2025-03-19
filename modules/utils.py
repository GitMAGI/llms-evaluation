def padding_computing(width, kernel, stride):
    ss = range(0,width-kernel,stride)[-1]
    #print(f"first ss: {ss}")
    while ss+kernel < width:
        ss = ss+stride
    #print(f"last ss: {ss}")
    return ((ss+kernel)-width)

def get_convolution_layer_expected_output_info(input, kernel_shape, stride_shape, n_filters=10, padding='valid'):
    n_dims = len(input.shape[1:-1])
    input_shape = tuple(input.shape[1:])
    padding_shape = tuple([0] * n_dims)
    if padding.lower() != 'valid'.lower():
        padding_shape = tuple([padding_computing(input_shape[:-1][i], kernel_shape[i], stride_shape[i]) for i in range(0,len(stride_shape))])
    expected_output_shape = tuple([int((d-kernel_shape[i]+2*padding_shape[i])/stride_shape[i])+1 for i, d in enumerate(input_shape[:-1])]) + (n_filters,)
    #print(f'Expected output shape (Conv 1D): {expected_output_shape}')
    return expected_output_shape, padding_shape