#ifndef THREADPOOL_H_
#define THREADPOOL_H_

#include "simulation_settings.h"

#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <future>
#include <utility>
#include <memory>

class ThreadPool {
public:
    std::vector<std::thread> threads;        // Worker threads
    std::queue<std::function<void()>> tasks;  // Queue of tasks
    std::mutex queue_mutex;                  // Mutex to synchronize access to shared data
    std::condition_variable cv;              // Condition variable to signal changes in the state of the tasks queue 
    bool stop = false;                       // Flag to indicate the threadpool should stop

    ThreadPool(int num_threads) {
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([this] {
                while(true) {
                    std::function<void()> task;

                    {
                        // Lock the queue
                        std::unique_lock<std::mutex> lock(queue_mutex);

                        // Wait for a task or the stop flag
                        cv.wait(lock, [this] { return !tasks.empty() || stop; });

                        // Exit thread
                        if (stop && tasks.empty()) { return; }

                        // Get the next task from the queue
                        task = move(tasks.front());
                        tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        // Stop the threads
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }

        // Notify all threads
        cv.notify_all();

        for (auto& thread : threads) {
            thread.join();
        }
    }

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using return_type = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

        std::future<return_type> result = task->get_future();

        {
            std::unique_lock lock(queue_mutex);
            tasks.emplace([task]() { (*task)(); });
        }
        cv.notify_one();

        return result;
    }
};

#endif  // THREADPOOL_H_