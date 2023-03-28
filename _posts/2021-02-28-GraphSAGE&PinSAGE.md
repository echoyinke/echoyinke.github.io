---
layout: post
title: "GraphSAGE&PinSAGE"
subtitle: 'graphSAGE'
author: "YiKe"
header-style: text
tags:
- impl
---



本文主要讲解两种**图采样算法**。前面GCN讲解的文章中，我使用的图$G$节点个数非常少，然而在实际问题中，一张图可能节点非常多，因此就没有办法一次性把整张图送入计算资源，所以我们应该使用一种有效的采样算法，从全图$G$中采样出一个子图$g$，这样就可以进行训练了

在了解图采样算法前，我们至少应该保证采样后的子图是**连通**的。例如下图中，左边采样的子图就是连通的，右边的子图不是连通的

![](https://ae01.alicdn.com/kf/U01691aeab5d14dd68be637a27b3d6a9a6.jpg#shadow)

#### GraphSAGE (SAmple & aggreGatE)

GraphSAGE主要分两步：采样、聚合

**采样**的阶段首先选取一个点，然后随机选取这个点的一阶邻居，再以这些邻居为起点随机选择它们的一阶邻居。例如下图中，我们要预测0号节点，因此首先随机选择0号节点的一阶邻居2、4、5，然后**随机**选择2号节点的一阶邻居8、9；4号节点的一阶邻居11、12；5号节点的一阶邻居13、15

![](https://ae01.alicdn.com/kf/U88a905090c0849c5ac3f48882f51dc3e4.jpg#shadow)

**聚合**具体来说就是直接将子图从全图中抽离出来，从最边缘的节点开始，一层一层向里更新节点

![](https://ae01.alicdn.com/kf/U2f13e05520684c3f8fe6bc35fa7efa08h.jpg#shadow)

下图展示了邻居采样的优点，极大减少训练计算量这个是毋庸置疑的，泛化能力增强这个可能不太好理解，因为原本要更新一个节点需要它周围的所有邻居，而通过邻居采样之后，每个节点就不是由所有的邻居来更新它，而是部分邻居节点，所以具有比较强的泛化能力

![](https://ae01.alicdn.com/kf/U4d12a98ca4d3494abdcb4a72488ce508e.jpg#shadow)

#### PinSAGE

**采样时只能选取真实的邻居节点吗？**如果构建的是一个与虚拟邻居相连的子图有什么优点？PinSAGE算法将会给我们解答

PinSAGE算法通过多次随机游走，按游走经过的频率选取邻居，例如下面以0号节点作为起始，随机进行了4次游走

![](https://ae01.alicdn.com/kf/U4a39337caa364c189d0237cea182df94p.jpg#shadow)

其中5、10、11三个节点出现的频率最高，因此我们将这三个节点与0号节点相连，作为0号节点的虚拟邻居

![](https://ae01.alicdn.com/kf/Ufb6a4ca71afc45eb8db24cfc67bf3eeaz.jpg#shadow)

回到上述问题，采样时选取虚拟邻居有什么好处？可以快速获取远距离邻居的信息。实际上如果是按照GraphSAGE算法的方式生成子图，在聚合的过程中，非一阶邻居的信息可以通过消息传递逐渐传到中心，但是随着距离的增大，离中心越远的节点，其信息在传递过程中就越困难，甚至可能无法传递到；如果按照PinSAGE算法的方式生成子图，有一定的概率可以将非一阶邻居与中心直接相连，这样就可以快速聚合到多阶邻居的信息

#### Reference

+   [https://baidu-pgl.gz.bcebos.com/pgl-course/lesson\_4.pdf](https://baidu-pgl.gz.bcebos.com/pgl-course/lesson_4.pdf)