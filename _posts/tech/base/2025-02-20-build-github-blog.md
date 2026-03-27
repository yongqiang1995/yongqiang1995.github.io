---
layout: post
title: "如何 <code>零成本</code> 打造个人博客"
subtitle: "基于 Jekyll & GitHub Pages 搭建个人博客"
# date: 2025-02-21 10:00:00
author: "YongQiang"
header-style: text # 博客 title 如果有图像背景, 需要注释
hidden: false # 是否参与主页推荐
published: true # 控制是否被 Jekyll 渲染
catalog: true # 功能不明
ref_id: "how-to-build-self-blog-with-github"
tags:
  - git-pages
  - jekyll
  - 博客搭建
---

![GitHub Pages + Jekyll](https://jekyllrb.com/img/logo-2x.png)

## 选择 GitHub Pages + Jekyll 的理由

- 🆓 **零成本托管**: `GitHub` 提供免费域名 (`username.github.io`) 及服务器资源

- ⚡ **自动部署**: 提交 `Markdown` 文件自动生成静态网页

- 🔄 **版本控制**: 原生集成 `Git` 实现内容版本管理

- 📱 **移动友好**: `Jekyll` 默认支持响应式布局

效果展示:

![](/img/in-post/yongqiang's-blog-homepage.png)

## 十分钟搭建指南 (含避坑要点)

### 前置条件

1. 注册 [GitHub](https://github.com) 账号

2. 本地安装 [Git](https://git-scm.com/) 命令行工具, 配置 `git` 环境


### 步骤一: 创建专属仓库

- 新建仓库名**必须**为: `<username>.github.io`
  - 注意其中的 `<username>` 需要替换为你的 `gitHub` 用户名

- **可选** `Add a README file`

创建仓库示意图如下:

![](/img/in-post//create-repository.png)

先前我已经建立了名为 `yongqiang1995.github.io` 的仓库, 故而系统会提示仓库存在, 无法重新建立.

```js
The repository yongqiang1995.github.io already exists on this account.
```

仓库创建好了之后 `clone` 到本地, 可以先使用 `https` 免去配置 `ssh` 的麻烦.

### 步骤二: 使用 `YoQ Blog` 模板

将作者仓库中的所有文件拷贝到你的个人仓库内:

```shell
$ git clone https://github.com/yongqiang1995/yongqiang1995.github.io.git # 先克隆到本地
$ cp -r yongqiang1995.github.io/* <username>.github.io/ # 文件拷贝
```

然后将你仓库中的所有所有文件 `push` 到远程仓库:

```shell
$ git status # 查看 git 追踪的文件状态
$ git add . # 将所有修改全部保存在工作区
$ git commit -m "init repo" # 准备提交工作区数据
$ git push origin main -f # -f 是覆盖性提交, 初始时使用, 后续提交时慎用
```

关于 `git` 命令的基础操作, 不熟悉的同学可以[参考这里](https://www.runoob.com/git/git-basic-operations.html).

当文件成功 `push` 到远程仓库后, `github` 将会自动执行 `git pages` 部署, 在 `https://github.com/<username>/<username>.github.io/deployments` 下可以查看部署情况, 部署完毕即可通过域名 `<username>.github.io` 访问博客系统.

### 步骤二: 安装 Jekyll

> 为什么需要在本地安装 `Jekyll` 环境? 因为 `Jekyll` 支持我们本地调试博客, 避免不断将修改 `push` 到 `Github` 中, 造成 `commit` 记录混乱且复杂.

安装 `Jekyll` 之前需要先安装 `ruby`, 作者以 `MacOS` 操作系统为例进行说明, 其他操作系统会更简单一些.

本地安装 `Ruby ≥2.5` ([点击访问官方安装指南](https://www.ruby-lang.org/zh_cn/documentation/installation/))

`ruby` 安装完成后, 查看版本信息:

```sh
$ brew install ruby # 使用 MacOS brew 包管理工具安装 ruby
$ which ruby # 检查当前使用的 ruby 路径
# /usr/local/opt/ruby/bin/ruby
$ ruby -v # 查看 ruby 版本
$ ruby 3.2.2 (2023-03-30 revision e51014f9c0) [x86_64-darwin22]
```

如果遇到 `command not found: ruby`, 需要手动配置环境变量 (❗️注意替换版本)

```shell
echo 'export PATH="/usr/local/opt/ruby/bin:/usr/local/lib/ruby/gems/3.2.0/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc # 使环境变量生效, 如果使用的是 .bashrc, 请注意替换 
```

`ruby` 安装成功后, 开始安装 `jekyll` 和 `bundler`

```shell
$ gem install jekyll bundler
```

在你本地的 `github.io` 目录下, 启动 `Jekyll`

```shell
$ cd <username>.github.io/
$ jekyll s
```

如果启动 `jekyll s` 失败, 可能出现以下错误:

```shell
$ ~/w/yongqiang1995.github.io on main ❯ jekyll s

/Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:355:in `raise_not_found!': Could not find gem 'jekyll-paginate' in locally installed gems. (Bundler::GemNotFound)
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:455:in `block in prepare_dependencies'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:430:in `each'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:430:in `filter_map'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:430:in `prepare_dependencies'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:64:in `setup_solver'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/resolver.rb:29:in `start'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/definition.rb:738:in `start_resolution'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/definition.rb:346:in `resolve'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/definition.rb:645:in `materialize'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/definition.rb:237:in `specs'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/definition.rb:304:in `specs_for'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler/runtime.rb:18:in `setup'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/bundler-2.6.4/lib/bundler.rb:167:in `setup'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/jekyll-4.4.1/lib/jekyll/plugin_manager.rb:52:in `require_from_bundler'
from /Users/yongqiang/.local/share/gem/ruby/3.2.0/gems/jekyll-4.4.1/exe/jekyll:11:in `<top (required)>'
from /usr/local/lib/ruby/gems/3.2.0/bin/jekyll:25:in `load'
from /usr/local/lib/ruby/gems/3.2.0/bin/jekyll:25:in `<main>'
```

尝试修复, 首先运行以下命令, 修复捆绑程序路径配置:

```sh
$ bundle config set path 'vendor/cache'
```

然后再运行:

```shell
$ bundle exec jekyll serve
```

上面会打印许多缺失的 package, 可以通过下面命令自动安装:

```shell
$ bundle install
```

最后需要更新 `bundle`,  确保 `Gemfile.lock` 引用最新的 `gem`:

```shell
$ bundle update
```

通过执行以上命令, `jekyll serve` 应该可以成功启动. 启动后日志如下: 

```shell
$ jekyll s
Configuration file: /Users/yongqiang/workspace/yongqiang1995.github.io/_config.yml
            Source: /Users/yongqiang/workspace/yongqiang1995.github.io
       Destination: /Users/yongqiang/workspace/yongqiang1995.github.io/_site
 Incremental build: disabled. Enable with --incremental
      Generating...
                    done in 2.48 seconds.
 Auto-regeneration: enabled for '/Users/yongqiang/workspace/yongqiang1995.github.io'
    Server address: http://127.0.0.1:4000/
  Server running... press ctrl-c to stop.
```

可以发现 `jekyll` 提供给我们了一个本地 `url` (`localhost: 127.0.0.1`), 端口在 `4000`, 点击访问就可以打开我们本地部署的博客系统了.

### 其他依赖

某些博客系统会依赖 `webrick` 组件, 但是目前该组件已经不属于 `ruby 3.0` 中的捆绑 `gem` 了, 需要手动添加 `webrick` 到 `Gemfile` 中作为依赖:

```shell
$ bundle add webrick
```

本博客原作者应是 [Hux-Blog](https://github.com/Huxpro/huxpro.github.io), 在它的基础上, 我补充了图像点击放大的功能, 十分感谢原作者提供的模板.

References
----------

- [icons图标下载](https://igoutu.cn/icons/set/github)
- [github-pages](https://pages.github.com/)
- [Jekyll如何实现图片点击放大](https://blog.walterlv.com/post/create-click-to-zoom-image-for-web-pages.html)
- [Hux-Blog](https://github.com/Huxpro/huxpro.github.io)
