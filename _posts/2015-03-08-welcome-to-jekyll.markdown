---
layout: post
title:  "Welcome to Jekyll!"
date:   2015-03-08 22:21:49
categories: Coding
tags: Post
---
You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll’s dedicated Help repository][jekyll-help].

## 图片

为了在`.md`中使用图片相对路径，保持本地和线上显示一致，需要修改Jekyll默认的永久链接格式。Jekyll默认的永久链接格式为：

```
permalink: :year/:month/:day/:title/
```

会导致使用相对路径插入图片的帖子（如`./assets/img/postsimg/20200505/1.jpg`）无法定位到图片的真正位置。因此需在`_config.yml`中修改为以下语句：

```json
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