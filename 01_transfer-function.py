import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# ===================================================
# CONFIG
# ===================================================

CONFIG = {
    "use_noise": True,
    "input_audio": "trumpet_open.wav",
    "target_audio": "trumpet_muted.wav",
    "filter_length": 128,
    "num_steps": 500,
    "lr": 1e-2,
    "noise_length": 16000,
}


# ===================================================
# DATA LOADING
# ===================================================

def make_bandpass_fir(L=128, sr=16000, f_low=2000, f_high=5000):

    # frequency axis
    freqs = torch.fft.rfftfreq(L, d=1/sr)

    # rectangular bandpass in frequency domain
    H = torch.zeros(len(freqs))

    H[(freqs >= f_low) & (freqs <= f_high)] = 1.0

    # random phase (important for realism)
    phase = torch.exp(1j * 2 * torch.pi * torch.rand_like(H))
    H_complex = H * phase

    # inverse FFT -> impulse response
    h = torch.fft.irfft(H_complex, n=L)

    # normalize
    h = h / h.abs().max()

    return h.view(1, 1, -1).float()


def load_audio_mode(config):
    """Returns x, y_target, sample_rate"""

    if config["use_noise"]:
        print("Using synthetic noise mode")

        torch.manual_seed(0)

        x = torch.randn(1, 1, config["noise_length"])

        h_true = make_bandpass_fir(
            L=128,
            sr=16000,
            f_low=2000,
            f_high=5000
        )

        y_target = F.conv1d(
            x,
            h_true,
            padding=h_true.shape[-1] // 2
        )

        y_target += 0.01 * torch.randn_like(y_target)

        return x, y_target, 16000, h_true

    else:
        print("Using real audio mode")

        x, sr = torchaudio.load(config["input_audio"])
        y_target, sr2 = torchaudio.load(config["target_audio"])

        assert sr == sr2

        x = x[:1]
        y_target = y_target[:1]

        min_len = min(x.shape[-1], y_target.shape[-1])
        x = x[:, :min_len]
        y_target = y_target[:, :min_len]

        x = x / x.abs().max()
        y_target = y_target / y_target.abs().max()

        x = x.unsqueeze(0)
        y_target = y_target.unsqueeze(0)

        return x, y_target, sr, None


# ===================================================
# TRAINING
# ===================================================

def train_filter(x, y_target, config):
    """Learns FIR filter via gradient descent"""

    L = config["filter_length"]

    h = torch.randn(1, 1, L, requires_grad=True)
    opt = torch.optim.Adam([h], lr=config["lr"])

    losses = []
    filters = []
    gradients = []
    spectra_pred = []

    target_fft = torch.fft.rfft(y_target[0, 0]).abs().detach().cpu().numpy()

    for step in range(config["num_steps"]):
        y_pred = F.conv1d(x, h, padding=L // 2)

        # FIX: match length exactly
        min_len = min(y_pred.shape[-1], y_target.shape[-1])

        y_target_crop = y_target[..., :min_len]
        y_pred = y_pred[..., :min_len]

        loss = torch.mean((y_pred - y_target_crop) ** 2)

        opt.zero_grad()
        loss.backward()

        losses.append(loss.item())

        filters.append(h.detach().cpu().numpy().squeeze().copy())
        gradients.append(h.grad.detach().cpu().numpy().squeeze().copy())

        spectra_pred.append(
            torch.fft.rfft(y_pred[0, 0]).abs().detach().cpu().numpy()
        )

        opt.step()

        if step % 50 == 0:
            print(f"step={step} | loss={loss.item():.6f}")

    return h, losses, filters, gradients, spectra_pred, target_fft


# ===================================================
# VISUALIZATION
# ===================================================

def run_animation(
    config,
    h,
    losses,
    filters,
    gradients,
    spectra_pred,
    target_fft
):

    L = config["filter_length"]
    num_steps = config["num_steps"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    ax_f, ax_g, ax_l, ax_s = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    line_f, = ax_f.plot([])
    line_g, = ax_g.plot([])
    line_l, = ax_l.plot([])
    line_s, = ax_s.plot([], [], label="Pred")
    ax_s.plot(target_fft, label="Target")

    ax_f.set_title("FIR Filter")
    ax_g.set_title("Gradient")
    ax_l.set_title("Loss")
    ax_s.set_title("Spectrum")
    ax_s.legend()

    ax_f.set_xlim(0, L)
    ax_g.set_xlim(0, L)
    ax_l.set_xlim(0, num_steps)

    ax_l.set_yscale("linear")

    def update(i):

        line_f.set_data(np.arange(L), filters[i])
        line_g.set_data(np.arange(L), gradients[i])
        line_l.set_data(np.arange(i+1), losses[:i+1])

        line_s.set_data(
            np.arange(len(spectra_pred[i])),
            spectra_pred[i]
        )

        ax_f.set_ylim(np.min(filters), np.max(filters))
        ax_g.set_ylim(np.min(gradients), np.max(gradients))

        ax_s.set_xlim(0, len(target_fft))
        ax_s.set_ylim(0, max(np.max(target_fft), np.max(spectra_pred[i])))

        return line_f, line_g, line_l, line_s

    ani = FuncAnimation(fig, update, frames=num_steps, interval=50, blit=False)

    plt.tight_layout()
    plt.show()

    return ani


# ===================================================
# MAIN
# ===================================================

def main():

    config = CONFIG

    x, y_target, sr, h_true = load_audio_mode(config)

    h, losses, filters, gradients, spectra_pred, target_fft = train_filter(
        x, y_target, config
    )

    ani = run_animation(
        config,
        h,
        losses,
        filters,
        gradients,
        spectra_pred,
        target_fft
    )

    plt.figure()
    plt.plot(losses)
    plt.title("Loss over time")
    plt.xlabel("Step")
    plt.ylabel("MSE Loss")
    plt.show()

    # optional: show true filter (noise mode)
    if h_true is not None:
        plt.figure()

        plt.plot(h_true.squeeze(), label="True")
        plt.plot(h.detach().squeeze(), label="Learned")

        plt.legend()
        plt.title("True vs Learned Filter")

        plt.show()


if __name__ == "__main__":
    main()
