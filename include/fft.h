// Enhanced FFT Header File
// Date: 2026-04-08
// Author: pansychen1980

#ifndef FFT_H
#define FFT_H

#include <complex>
#include <vector>

// Function to perform FFT on complex data
void fft(std::vector<std::complex<double>>& data);

// Function to extract magnitude from complex data
std::vector<double> extract_magnitude(const std::vector<std::complex<double>>& data);

// Function to extract phase from complex data
std::vector<double> extract_phase(const std::vector<std::complex<double>>& data);

// Function to zero-pad the data to the next power of two
std::vector<std::complex<double>> zero_pad(const std::vector<std::complex<double>>& data);

#endif // FFT_H