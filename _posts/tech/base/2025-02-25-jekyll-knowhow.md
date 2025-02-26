---
layout: post
title: "Jekyll Know-How"
# subtitle: "based Jekyll & github-pages"
# date: 2025-02-25 10:00:00
author: "YongQiang"
header-style: text # 博客 title 如果有图像背景, 需要注释
hidden: false # 隐藏博文?
published: true # 控制是否被 Jekyll 渲染
catalog: true # 功能不明
tags:
  - 博客开发
  - Jekyll
---

> 本节内容主要分享 `Jekyll` 的操作技巧, 例如不同 `web` 引用、不同的图像展示方法等, 代码不同, 风格各异.

### 网页内嵌方法: 显示部分博文区域并点击跳转

> 当你想要在当前文章中以图形图像的方式引用之前的博文, 并且可以触发点击跳转到原博文的功能, 可以参考以下实现.

- 📌 内嵌网页有一个边框, 且居中展示
- 📌 点击 `iframe` 直接跳转到原博客
- 📌 支持相对路径或参数化 `URL`, 适用于不同博客文章


以如何 「`自定义`」个人博客为例 (`点击图像自动跳转`):

{% assign blog_url = "/2025/02/25/dev-blog/" %}

<div style="display: flex; justify-content: center; margin: 20px 0;">
    <div style="position: relative; border: 2px solid #ddd; border-radius: 10px; padding: 10px; 
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); display: inline-block;">
        <iframe id="blog-frame" 
            src="{{ site.baseurl }}{{ blog_url }}"
            width="650px" height="300px" frameborder="0"></iframe>
        <div onclick="window.location.href=document.getElementById('blog-frame').src" 
            style="position: absolute; top: 10px; left: 10px; width: calc(100% - 20px); height: calc(100% - 20px); 
                   cursor: pointer; background: transparent;">
        </div>
    </div>
</div>

### 网页内嵌方法: 显示部分博文区域并滚动查看

> 当你想要在当前文章中以图形图像的方式引用之前的博文, 并且可以通过滑动鼠标来浏览原博文的功能, 可以参考以下实现.

- ✅ 内嵌网页有一个边框, 且居中展示
- ✅ 在 `iframe` 区域滑动鼠标查看博文, 同样可以操作对应页面
- ✅ 支持相对路径或参数化 `URL`, 适用于不同博客文章

以如何 「`零成本`」打造个人博客为例 (`在图像区域滚动浏览`):

{% assign blog_url = "/2025/02/20/build-github-blog/" %}
<div style="display: flex; justify-content: center; margin: 20px 0;">
    <div style="position: relative; border: 2px solid #ddd; border-radius: 10px; padding: 10px; 
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); display: inline-block;">
        <iframe id="blog-frame" 
            src="{{ site.baseurl }}{{ blog_url }}"
            width="650px" height="300px" frameborder="0"></iframe>
    </div>
</div>


### 图像内嵌: 带外框的图像样式

> 当你觉得单纯的贴一张图实在是太过单调, 想要给它加一个外框, 可以参考以下实现.

- 😊 内嵌图像有一个边框, 且居中展示
- 😊 点击图像即可放大, 鼠标悬停会显示 `title` 中的字样
- 😊 支持相对路径或参数化 `URL`, 适用于不同博客文章

代码如下 (只需要插入对应的 `.md` 文件, 设定图像路径即可):

{% assign img_url = "/img/in-post/kanam.jpg" %}
<div style="display: flex; justify-content: center; margin: 20px 0;">
    <div style="position: relative; border: 2px solid #ddd; border-radius: 10px; padding: 10px; 
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); display: inline-block;">
        <img src="{{img_url}}"
            alt="Kanam" 
            width="512px"
            height="512px"
            title="Kanam-Show">
    </div>
</div>

### 图像内嵌: 带外框和点击动画的图像样式

> 当你觉得图像在展示的时候只有一个外框依然很单调, 想要加一个动画效果, 可以参考以下实现.

- 😄 内嵌图像有一个边框, 且居中展示
- 😄 鼠标悬停便会出现旋转动画, 点击图像即可放大查看
- 😄 支持相对路径或参数化 `URL`, 适用于不同博客文章

代码如下 (只需要插入对应的 `.md` 文件, 设定图像路径即可):

{% assign img_url = "/img/in-post/kanam.jpg" %}

<style>
/* 新增动画样式 */
.img-hover-container:hover img {
    transform: scale(1.1) rotate(2deg);
    transition: all 0.3s ease-out;
    cursor: pointer;
}
</style>

<div style="display: flex; justify-content: center; margin: 20px 0;">
    <div class="img-hover-container" 
         style="position: relative; 
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 10px;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                display: inline-block;
                overflow: hidden; /* 防止放大溢出 */
                transition: transform 0.3s;">
        <img src="{{img_url}}"
            alt="Kanam" 
            style="width: 512px; 
                   height: 512px;
                   transition: inherit;"
            title="Kanam-Show">
    </div>
</div>

### 图像内嵌: 仅使用 img 标签的图像样式

> 当你觉得以上都太过于花哨, 只希望用 `img` 标签简单展示一幅图像, 可以参考以下实现.

- 😁 一幅普通图像, 会自动居中显示, 可以调节显示大小
- 😁 鼠标悬停便会出现旋转动画, 点击图像即可放大查看
- 😁 支持相对路径或参数化 `URL`, 适用于不同博客文章

代码如下 (只需要插入对应的 `.md` 文件, 设定图像路径即可):

{% assign img_url = "/img/in-post/kanam.jpg" %}
<img src="{{img_url}}"
    alt="Kanam" 
    width="512px"
    height="512px"
    style="border: 2px solid #eee"
    title="Kanam-Show">


### 图像内嵌: Markdown 的图像样式

> 当你只会使用 `Markdown` 语法进行图像展示, 可以参考以下实现.

- 😂 一幅普通图像, 全屏显示
- 😂 鼠标悬停也不会有任何文字提示, 点击图像即可放大查看你写入 `![text]` 中的文本内容
- 😂 支持相对路径或参数化 `URL`, 适用于不同博客文章

代码如下 (只需要插入对应的 `.md` 文件, 设定图像路径即可):

![Kanam](/img/in-post/kanam.jpg)

---

🎯🎯🎯 `Jekyll Know-How` 会持续更新🔥🔥🔥. 如果对博文有疑问, 就请评论说明吧! 😄😄