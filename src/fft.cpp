#include "fft.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

const double PI = 3.141592653589793238462643383279502884197;

// Helper: Check if number is power of 2
bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// Helper: Find next power of 2
size_t next_power_of_two(size_t n) {
    if (is_power_of_two(n)) return n;
    size_t power = 1;
    while (power < n) power *= 2;
    return power;
}

// Bit reversal permutation
void bit_reverse(std::vector<Complex>& data) {
    size_t N = data.size();
    size_t j = 0;
    for (size_t i = 0; i < N - 1; ++i) {
        if (i < j) std::swap(data[i], data[j]);
        size_t mask = N >> 1;
        while (j & mask) {
            j &= ~mask;
            mask >>= 1;
        }
        j |= mask;
    }
}

// Cooley-Tukey FFT Algorithm
void fft(std::vector<Complex>& data) {
    size_t N = data.size();
    if (N == 1) return;
    if (!is_power_of_two(N)) {
        throw std::invalid_argument("Data size must be a power of 2. Use zero_pad() first.");
    }
    
    bit_reverse(data);
    
    for (size_t s = 1; s <= static_cast<size_t>(log2(N)); ++s) {
        size_t m = 1 << s;
        Complex w_m = std::exp(Complex(0, -2.0 * PI / m));
        
        for (size_t k = 0; k < N; k += m) {
            Complex w(1, 0);
            for (size_t j = 0; j < m / 2; ++j) {
                Complex u = data[k + j];
                Complex v = data[k + j + m / 2] * w;
                data[k + j] = u + v;
                data[k + j + m / 2] = u - v;
                w *= w_m;
            }
        }
    }
}

// Inverse FFT
void ifft(std::vector<Complex>& data) {
    size_t N = data.size();
    if (!is_power_of_two(N)) {
        throw std::invalid_argument("Data size must be a power of 2");
    }
    
    for (auto& val : data) val = std::conj(val);
    fft(data);
    for (auto& val : data) val = std::conj(val) / static_cast<double>(N);
}

// Extract magnitude
std::vector<double> extract_magnitude(const std::vector<Complex>& data) {
    std::vector<double> magnitude(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        magnitude[i] = std::abs(data[i]);
    }
    return magnitude;
}

// Extract phase (in radians)
std::vector<double> extract_phase(const std::vector<Complex>& data) {
    std::vector<double> phase(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        phase[i] = std::arg(data[i]);
    }
    return phase;
}

// Extract both magnitude and phase
std::pair<std::vector<double>, std::vector<double>> extract_magnitude_and_phase(
    const std::vector<Complex>& data) {
    std::vector<double> magnitude(data.size());
    std::vector<double> phase(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        magnitude[i] = std::abs(data[i]);
        phase[i] = std::arg(data[i]);
    }
    return {magnitude, phase};
}

// Reconstruct from magnitude and phase
std::vector<Complex> from_magnitude_and_phase(
    const std::vector<double>& magnitude,
    const std::vector<double>& phase) {
    if (magnitude.size() != phase.size()) {
        throw std::invalid_argument("Magnitude and phase must have same size");
    }
    std::vector<Complex> result(magnitude.size());
    for (size_t i = 0; i < magnitude.size(); ++i) {
        result[i] = magnitude[i] * std::exp(Complex(0, phase[i]));
    }
    return result;
}

// Power spectrum
std::vector<double> get_power_spectrum(const std::vector<Complex>& data) {
    std::vector<double> power(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        double mag = std::abs(data[i]);
        power[i] = mag * mag;
    }
    return power;
}

// Zero-pad to next power of 2
std::vector<Complex> zero_pad(const std::vector<Complex>& data) {
    size_t original_size = data.size();
    size_t new_size = next_power_of_two(original_size);
    if (original_size == new_size) return data;
    std::vector<Complex> padded(new_size, Complex(0, 0));
    std::copy(data.begin(), data.end(), padded.begin());
    return padded;
}

// Zero-pad to specific size
std::vector<Complex> zero_pad_to_size(const std::vector<Complex>& data, size_t target_size) {
    if (target_size < data.size()) {
        throw std::invalid_argument("Target size must be >= original size");
    }
    std::vector<Complex> padded(target_size, Complex(0, 0));
    std::copy(data.begin(), data.end(), padded.begin());
    return padded;
}

// Remove padding
std::vector<Complex> unpad(const std::vector<Complex>& data, size_t original_size) {
    if (original_size > data.size()) {
        throw std::invalid_argument("Original size cannot exceed padded data size");
    }
    return std::vector<Complex>(data.begin(), data.begin() + original_size);
}

// Automatic FFT with zero-padding
std::vector<Complex> forward_auto(const std::vector<Complex>& data) {
    auto padded = zero_pad(data);
    fft(padded);
    return padded;
}

// Automatic inverse FFT with zero-padding
std::vector<Complex> inverse_auto(const std::vector<Complex>& data, size_t original_size) {
    auto padded = zero_pad_to_size(data, next_power_of_two(original_size));
    ifft(padded);
    return unpad(padded, original_size);
}

// DC shift 1D
std::vector<Complex> dc_shift(const std::vector<Complex>& data) {
    std::vector<Complex> shifted = data;
    size_t N = data.size();
    for (size_t i = 0; i < N; ++i) {
        if ((i & 1) == 0) shifted[i] = -shifted[i];
    }
    return shifted;
}

// DC shift 2D
std::vector<std::vector<Complex>> dc_shift_2d(const std::vector<std::vector<Complex>>& data) {
    size_t rows = data.size();
    if (rows == 0) return data;
    size_t cols = data[0].size();
    std::vector<std::vector<Complex>> shifted = data;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (((i + j) & 1) == 0) shifted[i][j] = -shifted[i][j];
        }
    }
    return shifted;
}

// 2D FFT
std::vector<std::vector<Complex>> fft_2d(const std::vector<std::vector<Complex>>& data) {
    size_t rows = data.size();
    if (rows == 0) return data;
    size_t cols = data[0].size();
    if (!is_power_of_two(rows) || !is_power_of_two(cols)) {
        throw std::invalid_argument("Both dimensions must be powers of 2");
    }
    
    std::vector<std::vector<Complex>> result = data;
    
    for (size_t i = 0; i < rows; ++i) {
        std::vector<Complex> row = result[i];
        fft(row);
        result[i] = row;
    }
    
    for (size_t j = 0; j < cols; ++j) {
        std::vector<Complex> col(rows);
        for (size_t i = 0; i < rows; ++i) col[i] = result[i][j];
        fft(col);
        for (size_t i = 0; i < rows; ++i) result[i][j] = col[i];
    }
    
    return result;
}

// 2D Inverse FFT
std::vector<std::vector<Complex>> ifft_2d(const std::vector<std::vector<Complex>>& data) {
    size_t rows = data.size();
    if (rows == 0) return data;
    size_t cols = data[0].size();
    if (!is_power_of_two(rows) || !is_power_of_two(cols)) {
        throw std::invalid_argument("Both dimensions must be powers of 2");
    }
    
    std::vector<std::vector<Complex>> result = data;
    
    for (size_t i = 0; i < rows; ++i) {
        std::vector<Complex> row = result[i];
        ifft(row);
        result[i] = row;
    }
    
    for (size_t j = 0; j < cols; ++j) {
        std::vector<Complex> col(rows);
        for (size_t i = 0; i < rows; ++i) col[i] = result[i][j];
        ifft(col);
        for (size_t i = 0; i < rows; ++i) result[i][j] = col[i];
    }
    
    return result;
}

// Automatic 2D FFT with padding
std::vector<std::vector<Complex>> forward_auto_2d(const std::vector<std::vector<Complex>>& data) {
    auto padded = zero_pad_2d(data);
    return fft_2d(padded);
}

// 2D zero-pad
std::vector<std::vector<Complex>> zero_pad_2d(const std::vector<std::vector<Complex>>& data) {
    size_t rows = data.size();
    if (rows == 0) return data;
    size_t cols = data[0].size();
    size_t new_rows = next_power_of_two(rows);
    size_t new_cols = next_power_of_two(cols);
    
    std::vector<std::vector<Complex>> padded(new_rows, std::vector<Complex>(new_cols, Complex(0, 0)));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            padded[i][j] = data[i][j];
        }
    }
    return padded;
}

// 2D magnitude extraction
std::vector<std::vector<double>> get_magnitude_2d(const std::vector<std::vector<Complex>>& data) {
    size_t rows = data.size();
    if (rows == 0) return {};
    size_t cols = data[0].size();
    std::vector<std::vector<double>> magnitude(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            magnitude[i][j] = std::abs(data[i][j]);
        }
    }
    return magnitude;
}

// Logarithmic scaling
std::vector<double> log_scale(const std::vector<double>& data, double offset) {
    std::vector<double> scaled(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        scaled[i] = log(std::abs(data[i]) + offset);
    }
    return scaled;
}

// 2D Logarithmic scaling
std::vector<std::vector<double>> log_scale_2d(const std::vector<std::vector<double>>& data, double offset) {
    size_t rows = data.size();
    if (rows == 0) return data;
    size_t cols = data[0].size();
    std::vector<std::vector<double>> scaled(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            scaled[i][j] = log(std::abs(data[i][j]) + offset);
        }
    }
    return scaled;
}