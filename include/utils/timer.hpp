#include <chrono>
#include <iostream>

class Timer {
public:
    Timer(std::string name = "none") : 
    name_(name) 
    { 
        start_ = std::chrono::steady_clock::now();
    }

    ~Timer()
    {
        end_ = std::chrono::steady_clock::now();
        duration_ = end_ - start_;
        double ms = duration_.count() * 1000.0;
        std::cout << name_ << " costs: " << ms << "ms" << std::endl;
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start_, end_;
    std::chrono::duration<double> duration_;
    std::string name_;
};