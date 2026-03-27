---
layout: post
title: 本地部署 RAG-LLM 模型 
author: "YongQiang"
header-style: text
hidden: false
published: false
tags:
  - 模型部署
  - Dify
---

## ollama

```
ollama run deepseek-r1:7b 
```

## Dify



```sh
git clone https://github.com/langgenius/dify.git
cd dify/docker
cp .env.example .env
docker compose up -d # 如果版本是 Docker Compose V1，使用以下命令：docker-compose up -d
```
Dify 社区版默认使用 80 端口，点击链接 http://your_server_ip 即可访问你的私有化 Dify 平台。


## 将本地 image 拷贝到远程服务器

```sh
$ docker images --format "{{.Repository}}:{{.Tag}} | {{.ID}}" | column -t -s "|"

ubuntu/squid:latest                          dae40da440fe
langgenius/dify-plugin-daemon:0.0.3-local    98e493768cf1
langgenius/dify-api:1.0.0                    e75fe639e420
langgenius/dify-web:1.0.0                    0325f9a1e172
postgres:15-alpine                           afbf3abf6aeb
nginx:latest                                 b52e0b094bc0
redis:6-alpine                               6dd588768b9b
langgenius/dify-sandbox:0.2.10               4328059557e8
semitechnologies/weaviate:1.19.0             8ec9f084ab23
nginx:profiling                              de18c5524b06
pulsar/mlir_v005:latest                      92cf57937a25
gitlab/gitlab-ce:latest                      7b8db4329c1c
portainer/portainer:latest                   580c0e4e98b0
docker101tutorial:latest                     e6bc19a3d687
ubuntu:latest                                4e2eef94cd6b
```

输出

docker 批量打包镜像, 在远程服务器 `docker load -i combined_imgs.tar` 即可.



## Reference

- <https://github.com/langgenius/dify>
- [Dify接入Deepseek](https://docs.dify.ai/zh-hans/learn-more/use-cases/private-ai-ollama-deepseek-dify)


## Docker 容器问题

由于服务器 docker 容器内的用户命名空间是一个数字用户, 会导致一些权限问题 (如果在容器内创建与宿主机 UID 对应的用户, 应该能解决这个问题)

查看 nginx 容器状态

```bash
   # 检查容器状态
   docker inspect docker-nginx-1 | grep -i status

   # 访问服务验证
   docker exec -it docker-nginx-1 curl -I http://localhost:80
```

进入 docker 容器内排查问题

```sh
# 临时进入容器检查配置
docker exec -it 0da7d271e87b bash
```

获取容器的日志:

```bash
# 查看所有容器的日志（按需替换容器名）
docker logs <CONTAINER_NAME> --tail=100
```

查看容器内 PostgreSQL 用户的 UID​

```bash
docker run --rm postgres:15-alpine id -u postgres
```

ssh 代理, 访问服务器 web


先不在 A100 上搞 Dify 了
