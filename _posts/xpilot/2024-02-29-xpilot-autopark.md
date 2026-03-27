---
layout: post
title: 自动泊车方案调研
author: "YongQiang"
header-style: text
hidden: true
published: false
tags:
  - XPilot
  - BEV
---

## APA 自动泊车主要功能

- 车位检测 (Detect Parking Slot)
- 泊车路径规划 (Path Planning)
- 轨迹追踪控制 (Controling)

## 常见的泊车方式

- 平行式泊车 (侧方停车)
- 垂直式泊车 (倒车入库)
- 斜列式泊车 (斜列式的停车位)

![alt text](/img/in-post/post-xpilot-autopark/image.png):

- (a) (b) (c) and (d) 车位是垂直的
- (e), (f) and (g) 车位是平行的(相对于车)
- (h) 是倾斜的


## 泊车技术等级分类

- AVP (Automated/Autonomous Valet Parking 自动/自主代客泊车), L4 高级自动泊车技术, 不需要人工干预
- HPA 自学习型泊车, L3
- RPA 遥控泊车, L2 狭义上的遥控泊车, 初衷是为了解决车辆入库后驾驶员无法打开车门的问题
- APA 全自动泊车, L2 低速、驾驶员不能离开舱位, 以便随时接管车辆

![alt text](/img/in-post/post-xpilot-autopark/image-1.png)

- A*
- hybrid-A*
- 在应对典型的如垂直泊入和水平泊入问题, 目前量产的方案大都是基于直线圆弧的几何连接算法 https://zhuanlan.zhihu.com/p/548096462

自动泊车路径规划方法大致可以分为以下 4 类:

- 几何法
  - 几何法大多考虑采用圆弧加直线来构造路径, 但是采用该法构造的路径曲率不连续, 导致车辆在进行路径跟踪时出现原地转向问题, 并且该方法对于泊车初始位姿要求比较严格

- 多步法
  - 多步法是将自动泊车路径约束进行逐步考虑, 然后分步获取满足约束的路径

- 采样法
  - 基于采样的寻径算法

- 数值优化法
  - 引入最优目标, 将非完整系统的运动规划问题重新表述为开环最优控制问题, 并进行数值求解


## AX-APA 泊车简易 Demo (L2)


模型: dmpr_lse

数据集: ps2.0

局部路径规划方法: 多项式曲线法-五次多项式

耗时记录: 

![alt text](/img/in-post/post-xpilot-autopark/image-2.png)

### Tesla Model Y 的自动泊车限制

ref: https://www.tesla.com/ownersmanual/modely/zh_cn/GUID-0C763E08-D0B8-4404-8180-1054F635C08C.html



### 纵目科技 AVP 智能驾驶 DEMO

- 一汽红旗 demo 视频
- 长安 demo 视频

### KS TDA4 芯片部署

> 多个组合作, 工期花费 4 个月

视频省略.





