# 导师任务分配


要求实现一篇论文，论文信息：“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts, ACM SIGGRAPH, 2004. https://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf 


### 实现GrabCut任务时请注意：
1.  Matting部分不是GrabCut算法的精华，这部分可以不做。
2.  注意输出整个程序的一些重要中间结果（那些重要自己判断），面试中会有相关问题。
3.  如果有期末考试之类的事情，优先准备期末考试，不用急着参与测验。测验问题回答时的错误不要归因于忙期末考试所以没时间细看论文。 
4.  面试用QQ视频共享桌面的形式。
5.  代码请用C++自己实现（GraphCut部分可以调用现成的函数库：https://vision.cs.uwaterloo.ca/code/ 中的Max-flow/min-cut）。图片的输入输出和用户交互部分推荐调用OpenCV。Python 语言不是不让用，但是很难写出实时运行的代码
6.  完成这个任务时，请思考一个问题，GMM颜色模型换成颜色直方图（可以参考这篇论文https://mmcheng.net/zh/salobj/ 看看怎么实现快速的彩色颜色的直方图），会对结果有什么影响。
7.  对于一个400*600的图像，这个程序运行时间通常是1s以内（实现的好的话，0.1s左右也很正常）。如果你的程序运行时间明显过长，请认真优化。注意测量程序执行时间请用release模式（显示，实际运行等场合用的），而不是debug模式（调试程序用的，经常比release模式慢10倍左右）。

## Reading Paper
From the paper, we need to understand the entire workflow of the whole grabcut procedure and the algorithm it used.  
After reading, Here are my brief conclusion: Given an RGB image, Firstly we need to seperated the roughly background and foreground by a rectangle which denote the initial foreground.  
Secondly, the author use GMM model to describe the background and foreground colour point cluster distribution. To reflect how the current segmentation is fit into the ground true, 
the paper gives an energy function, the penalties for assigning pixel to "foreground" or "background".  
Thirdly, for minimizing the energy, it generates a graph, where each pixel represent a node , and its edges reveal the energy cost.
It utilizes the min-cut algorithm to find a cut with the min energy value, and then use the cut generating new foreground and background, recalculating the GMM model's parameters, updating the graph's edges and doing this iteratively to
achieve the best result of segmentation.

the graph definition:  
![1.jpg](data%2F1.png)


## GMM model
A kind of cluster algorithm, here are some helpful articles to understand it.  
https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95
https://zhuanlan.zhihu.com/p/30483076

The paper use GMM model to calculate the uncertain pixels' probability to belong fgd or bgd, and set it to the t-links' weight(we will discuss t-link below)    
To successfully reimplemented the paper's work, we need to construct two GMM model for the bgd_points and fgd_points.  
For the GMM part, what we need to do cover:
1. init bgd and fgd GMM model's parameter(weight, means, covariance) by KMeans algorithm.
2. calculate the K value for each pixel in the whole image
3. update the bgd and fgd GMM model's parameter by the newly calculated K from step 2.

## Minimum Cut
Some helpful article:
https://www.baeldung.com/cs/minimum-cut-graphs#:~:text=The%20minimum%20cut%20of%20a%20weighted%20graph%20is%20defined%20as,the%20graph%20into%20two%20sets.

Minimum cut is an algorithm to separate a graph's nodes into to two categories, which have the minimum sum of cut edges' weight.  
The paper use the minimum cut algorithm as a tool to minimize the energy function. It sets each graph's node with two kind of edges, 
t-links(terminal-links) and n-links(neighbor-links), and use the U(region function) and V(boundary function) 
to get the weights of two links' value.  
After We define a graph and it's edges, we can get the current minimum cut, and get the region of new bgd and fgd.  
For the Minimum Cut part, what we need to do cover:
1. calculate the const beta and K value.
2. generate a graph by the given GMM model and bgd_fgd_distribution(mask).
3. do max-flow to get the minimum cut.

t-links and n-links definition:  
![2.png](data%2F2.png)

## Combine the GMM and Minimum cut
1. init GMM's and Minimum-cut's parameters.
2. Assign GMM components to pixels.
3. Learning GMM parameters from data.
4. Estimate segmentation.
5. Repeat from step2, until convergence. 

## install dependence
### install opencv
```
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip pkg-config
sudo apt-get install libavformat-dev libavcodec-dev libswscale-dev
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p build && cd build
# Configure
cmake -DWITH_FFMPEG=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build .
# Install 
sudo make install 
```