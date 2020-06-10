---
layout: post
title:  "欢迎使用Jekyll！"
date:   2015-03-08 22:21:49
categories: Coding
tags: Post
---

发布的帖子位于 `_posts` 目录。如果想新建帖子，在 `_posts` 目录中新建一个`.md`文件，文件名命名为 `YYYY-MM-DD-title` 的格式。

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll’s dedicated Help repository][jekyll-help].

## 图片

为了在`.md`中使用图片相对路径，保持本地和线上显示一致，需要修改Jekyll默认的永久链接格式。Jekyll默认的永久链接格式为：

```
permalink: :year/:month/:day/:title/
```

会导致使用相对路径插入图片的帖子（如`./assets/img/postsimg/20200505/1.jpg`）无法定位到图片的真正位置。因此需在`_config.yml`中修改为以下语句：

```
permalink: :year-:month-:day-:title/
```

修改后，`_posts`中帖子的图片无论是在本地还是在线上均能正常显示。

## 公式

Jekyll虽然支持Markdown，但是不能正确显示公式，可以借用MathJax帮助渲染。

方法：

- 设置markdown引擎为kramdown，方法为在 `_config.yml `里添加：`markdown: kramdown`

- 在md文件开始输入代码：

```html
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```

然后正文就可以写公式：`$ e = m c^2 $ `这样就能正确显示了。



[jekyll]:      http://jekyllrb.com
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-help]: https://github.com/jekyll/jekyll-help

Cugtyt. [让GitHub Page支持Latex公式](https://zhuanlan.zhihu.com/p/36302775)

知乎. [Jekyll博客中如何用相对路径来加载图片？](https://www.zhihu.com/question/31123165?sort=created)