#ifndef PROGRESS_BAR_H
#define PROGRESS_BAR_H

#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cstdint>

namespace progress {

class ProgressBar {
private:
    std::atomic<int> counter_;              // Current iteration count (atomic for thread-safety)
    int total_;                             // Total iterations
    std::string description_;               // Progress bar description
    bool disabled_;                         // If true, all operations are no-ops
    int max_refresh_rate_;                  // Max refreshes per second (Hz)
    mutable std::mutex mutex_;              // Mutex for thread-safe operations
    std::atomic<bool> started_;             // Whether the timer has started
    std::atomic<int64_t> start_time_ms_;    // Start time in milliseconds
    std::atomic<int64_t> last_refresh_ms_;  // Last refresh time in milliseconds
    int bar_width_;                         // Width of the progress bar (default 60)
    int last_output_length_;                // Length of last output for proper clearing

    static int64_t now_ms() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    }

    void start_if_needed() {
        if (started_.load(std::memory_order_acquire)) return;
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_.load(std::memory_order_relaxed)) {
            int64_t now = now_ms();
            start_time_ms_.store(now, std::memory_order_relaxed);
            last_refresh_ms_.store(now, std::memory_order_relaxed);
            started_.store(true, std::memory_order_release);
        }
    }

    // Calculate elapsed time in seconds
    double elapsed_time() const {
        if (!started_.load(std::memory_order_acquire)) return 0.0;
        int64_t now = now_ms();
        int64_t start_ms = start_time_ms_.load(std::memory_order_relaxed);
        return (now - start_ms) / 1000.0;
    }

    // Check if enough time has passed to refresh (inline for performance)
    inline bool should_refresh() {
        if (disabled_ || max_refresh_rate_ <= 0) [[unlikely]] return false;
        if (!started_.load(std::memory_order_acquire)) return false;

        int64_t now = now_ms();
        int64_t last_refresh = last_refresh_ms_.load(std::memory_order_relaxed);
        int64_t time_since_refresh = now - last_refresh;
        // Convert Hz to milliseconds: 1000ms / refresh_rate (e.g., 5 Hz = 200ms)
        int64_t min_interval = std::max<int64_t>(1, 1000 / max_refresh_rate_);
        return time_since_refresh >= min_interval;
    }

    // Format time duration as MM:SS
    std::string format_time(double seconds) const {
        int mins = static_cast<int>(seconds) / 60;
        int secs = static_cast<int>(seconds) % 60;
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << mins << ":"
            << std::setfill('0') << std::setw(2) << secs;
        return oss.str();
    }

    // Build the progress bar string
    void render() {
        if (disabled_) return;

        int current = counter_.load(std::memory_order_relaxed);
        double elapsed = elapsed_time();
        double rate = (elapsed > 0) ? (current / elapsed) : 0.0;

        // Calculate percentage
        int percentage = (total_ > 0) ? static_cast<int>((current * 100.0) / total_) : 0;

        // Build bar
        int filled = (total_ > 0) ? static_cast<int>((current * bar_width_) / total_) : 0;
        filled = std::min(filled, bar_width_);

        std::string bar;
        bar.reserve(bar_width_ + 2);
        bar += "|";
        for (int i = 0; i < bar_width_; ++i) {
            if (i < filled) {
                bar += "#";
            } else {
                bar += " ";
            }
        }
        bar += "|";

        // Calculate ETA
        std::string eta_str;
        if (rate > 0.01 && current < total_ && current > 0) {
            double remaining = (total_ - current) / rate;
            // Only show ETA if it's reasonable (not too large)
            if (remaining < 86400) {  // Less than 24 hours
                eta_str = format_time(remaining);
            } else {
                eta_str = "--:--";
            }
        } else {
            eta_str = "--:--";
        }

        // Format output: description: percentage|bar| current/total [elapsed<eta, rate it/s]
        std::ostringstream oss;
        oss << description_ << ": "
            << std::setw(3) << percentage << "%"
            << bar << " "
            << current << "/" << total_ << " "
            << "[" << format_time(elapsed) << "<" << eta_str << ", "
            << std::fixed << std::setprecision(2) << rate << "it/s]";

        std::string output = oss.str();
        int output_length = static_cast<int>(output.length());

        // Clear previous line if it was longer
        std::cerr << "\r";
        if (output_length < last_output_length_) {
            std::cerr << std::string(last_output_length_, ' ') << "\r";
        }

        // Write new output
        std::cerr << output << std::flush;
        last_output_length_ = output_length;
    }

public:
    // Constructor
    ProgressBar(int total = 0, bool disabled = false, int max_refresh_rate = 5, int bar_width = 60)
        : counter_(0),
          total_(total),
          description_("Progress"),
          disabled_(disabled),
          max_refresh_rate_(max_refresh_rate),
          started_(false),
          start_time_ms_(0),
          last_refresh_ms_(0),
          bar_width_(bar_width),
          last_output_length_(0) {
    }

    // Destructor
    ~ProgressBar() {
        // Nothing to do - close() should be called explicitly
    }

    // Set the description and refresh display
    void set_description(const std::string& desc) {
        if (disabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        description_ = desc;
        render();
        last_refresh_ms_.store(now_ms(), std::memory_order_relaxed);
    }

    // Set the counter to a specific value and refresh if enough time has passed
    void set_counter(int value) {
        if (disabled_) return;
        counter_.store(value, std::memory_order_relaxed);
        start_if_needed();

        // Refresh only if enough time has passed
        if (should_refresh()) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (should_refresh()) {
                render();
                last_refresh_ms_.store(now_ms(), std::memory_order_relaxed);
            }
        }
    }

    // Increment counter by n
    void increment(int n = 1) {
        if (disabled_) return;
        counter_.fetch_add(n, std::memory_order_relaxed);
        // Lazy initialization: only set start time once
        // Use relaxed memory order since exact timing doesn't matter for display
        start_if_needed();
    }

    // Update counter and refresh if needed (efficient for loops)
    void update(int n = 1) {
        increment(n);
        if (should_refresh()) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (should_refresh()) {
                render();
                last_refresh_ms_.store(now_ms(), std::memory_order_relaxed);
            }
        }
    }

    // Close the progress bar (print final state and newline)
    void close(const std::string& final_desc = "") {
        if (disabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        // Set final description if provided
        if (!final_desc.empty()) {
            description_ = final_desc;
        }
        // Set counter to total
        counter_.store(total_, std::memory_order_relaxed);

        // Render final state
        int current = total_;
        double elapsed = elapsed_time();
        double rate = (elapsed > 0) ? (current / elapsed) : 0.0;

        // Build final bar (100%)
        std::string bar = "|";
        for (int i = 0; i < bar_width_; ++i) {
            bar += "#";
        }
        bar += "|";

        // Format final output without ETA (since we're done)
        std::ostringstream oss;
        oss << description_ << ": 100%" << bar << " "
            << current << "/" << total_ << " "
            << "[" << format_time(elapsed) << ", "
            << std::fixed << std::setprecision(2) << rate << "it/s]";

        std::string output = oss.str();
        int output_length = static_cast<int>(output.length());

        std::cerr << "\r";
        if (output_length < last_output_length_) {
            std::cerr << std::string(last_output_length_, ' ') << "\r";
        }
        std::cerr << output << std::endl;
    }

    // Get current counter value
    int get_counter() const {
        return counter_.load(std::memory_order_relaxed);
    }

    // Set total
    void set_total(int total) {
        if (disabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        total_ = total;
    }

    // Get total
    int get_total() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return total_;
    }
};

} // namespace progress

#endif // PROGRESS_BAR_H
