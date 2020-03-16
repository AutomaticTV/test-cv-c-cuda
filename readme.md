# Vision/C++/CUDA Test

## Exercise 1 [Multithreading C++]

Write a program in C++ using STL that allows the user to control multiple worker threads

###Requirements
Implement at least two distinct workers (e.g. computing some math function for a long time, writing and reading large files, etc.).
A worker thread must be safely paused, restarted, stopped or destroyed.
The program should accept the following arguments:
./program --help prints help message and instructions
./program --threads <nb_threads_to_run> starts <nb_threads_to_run> threads and wait for instructions. Each thread should have an id in the range [1, <nb_threads_to_run>]

Once the threads are started, the program should read instructions from the standard input. These instructions should have the following format:
pause <thread_id> pauses the thread with the given id and print a confirmation message
restart <thread_id> restarts the thread with the given id (if paused) and print a confirmation message
stop <thread_id> stops the thread with the given id (if not stopped) and print a confirmation message
status prints the id, the status (paused, running, stopped, finished) and the current processing step for each thread

The program should exit gracefully when all the worker threads are finished. Invalid program  options and instructions should be signaled.
The clarity and the extensibility of the solution is greatly appreciated.

## Exercise 2 [CUDA / Optimization]

We would like to optimize some CPU code which, given the last layer of a Deep Learning network, checks if each element validates some threshold and produces a DetectedObject element.
The output_ptr is a pointer to an image of size width x height, that can be assumed to be in GPU. 
We do not want you to write code, just to give an idea of how you would optimize this code.

std::vector<DetectedObject> postProcess(float *output_ptr, int width, int height, int batchSize, std::vector<std::vector<int>> anchors, int classes, std::vector<float> classThresh, int inputWidth, int inputHeight) {

    std::vector<DetectedObject> detections;
    assert(classThresh.size() == classes);
    int num_anchors = anchors.size();
    int num_channels = 5 + classes;

    for (int b = 0; b < batchSize; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int a = 0; a < num_anchors; ++a) {
                    float prob = sigmoid(get_item(output_ptr, b, a, 4, h, w, num_anchors, num_channels, height, width));

                    int max_id = 0;
                    float max_score = -1;
                    for (int c = 0; c < classes; ++c) {
                        float class_prob = sigmoid(get_item(output_ptr, b, a, 5 + c, h, w, num_anchors, num_channels, height, width));
                        float score = prob * class_prob;
                        if (max_score < score) {
                            max_id = c;
                            max_score = score;
                        }
                    }

                    if (max_score < classThresh[max_id]) {
                        continue;
                    }

                    float x = sigmoid(get_item(output_ptr, b, a, 0, h, w, num_anchors, num_channels, height, width));
                    float y = sigmoid(get_item(output_ptr, b, a, 1, h, w, num_anchors, num_channels, height, width));
                    float x_w = exponential(get_item(output_ptr, b, a, 2, h, w, num_anchors, num_channels, height, width)) * anchors[a][0];
                    float y_h = exponential(get_item(output_ptr, b, a, 3, h, w, num_anchors, num_channels, height, width)) * anchors[a][1];

                    x += w;
                    y += h;
                    x /= width;
                    y /= height;

                    x_w /= inputWidth;
                    y_h /= inputHeight;

                    x -= x_w / 2;
                    y -= y_h / 2;

                    cv::Rect_<float> bbox;
                    bbox.x = x;
                    bbox.y = y;
                    bbox.width = x_w;
                    bbox.height = y_h;
                    detections.emplace_back(max_id, max_score, bbox, b);
                }
            }
        }
    }
    return detections;
}

inline float get_item(float *output_data_ptr, int b, int a, int c, int h, int w, int num_anchors, int num_channels, int height, int width) {
    return output_data_ptr[b*(num_anchors * num_channels * height * width) + a * (num_channels * height * width) + c * (height * width) + h * width + w];
}

inline float sigmoid(float in) {
    return 1.f / (1.f + exp(-in));
}
inline float exponential(float in) {
    return exp(in);
}

