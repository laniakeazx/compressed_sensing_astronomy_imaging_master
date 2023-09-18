import numpy as np

def l1_norm(signal):
    return np.sum(np.abs(signal))
def sigma_mad(signal):
    return 1.4826 * np.median(np.abs(signal - np.median(signal)))
def fft(data):
    return (1 / np.sqrt(data.size) *
            np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(data))))
def ifft(data):
    return (np.sqrt(data.size) *
            np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(data))))

# normalised FFT
def nfft(data):
    return 1 / np.sqrt(data.size) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(data)))

# normalised inverse FFT
def infft(data):
    return np.sqrt(data.size) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(data)))

# Function to mask a signal.
def mask_op(signal, mask):

    return signal[np.where(mask == 1)[0]]

# Function to upsample a signal.
def upsample(signal, mask, dtype=complex):

    val = np.copy(mask).astype(dtype)
    val[val == 1] *= signal # val[val == 1] = val[val == 1] * signal
    return val

# Function for performing soft thresholding with complex coefficients.
def soft_thresh(data, threshold):

    return np.around(np.maximum((1.0 - threshold / np.maximum(np.finfo(np.float64).eps, np.abs(data))),0.0) * data,
                     decimals=15)

# Function to calculate the cost.
def cost_func(y, alpha_rec, mask, lambda_val):
    return (0.5 * np.linalg.norm(y - mask_op(nfft(alpha_rec), mask)) ** 2 + lambda_val * l1_norm(alpha_rec))


# Function that performs simple forward backward splitting.
def forwardBackward(observation, first_guess, mask, grad, lambda_val, n_iter=300, gamma=1.0, return_cost=False):
    alpha_rec = first_guess
    cost = []

    for i in range(n_iter):
        alpha_temp = alpha_rec - gamma * grad(observation, alpha_rec, mask)

        alpha_rec = soft_thresh(alpha_temp, lambda_val)
        cost.append(cost_func(observation, alpha_rec, mask, lambda_val))
    print("alpha_temp,grad", np.shape(alpha_temp), np.shape(grad))

    if return_cost:
        return alpha_rec, cost
    else:
        return alpha_rec

# Function to calculate the gradient
def grad(y, alpha_rec, mask):

    return infft(upsample(mask_op(nfft(alpha_rec), mask) - y, mask))

# Interactive function to plot the recovered alpha after a given number of iterations
def solve(n_iter):
    alpha_rec, cost = forwardBackward(observation=y, first_guess=np.ones(n_samples), mask=mask, grad=grad,
                                      lambda_val=0.01, n_iter=n_iter, gamma=1.0, return_cost=True)

    plot.stem_plot(alpha_rec, title=r'$\hat{\alpha}$')  # 时域待恢复信号

    if alpha_rec is not None:
        plot.stem_plot(nfft(alpha_rec), title=r'$\hat{\alpha_r}$')  # 频域信号

    plot.cost_plot(cost)

# normalised mean squared error
def nmse(signal_1, singal_2):
    return (np.linalg.norm(singal_2 - signal_1) ** 2 / np.linalg.norm(signal_1) ** 2)
