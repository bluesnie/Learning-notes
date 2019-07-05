// document.write('hello world');

window.onload = function () {  // window.onload() 方法用于在网页加载完毕后立刻执行的操作
    imgLocation("container", 'box');
    var imgData = {"data":[{"src":"../../res/waterfall1.jpg"},{"src":"../../res/waterfall2.jpg"},{"src":"../../res/waterfall3.jpg"},
            {"src":"../../res/waterfall4.jpg"},{"src":"../../res/waterfall5.jpg"}]};  // 模拟数据
    window.onscroll = function () {
        if(checkFlag()){                                                      //是否到底部
            var cparent = document.getElementById("container");    // 获取父级  (优化：因为很多地方用到可以分装成一个函数)
            for(var i=0;i<imgData.data.length;i++){
                var ccontent = document.createElement("div");       // 增加一个div
                ccontent.className = "box";                                   // div 类名为box
                cparent.appendChild(ccontent);                                // 父级上追加
                var boximg = document.createElement("div");         // 再添加一个div
                boximg.className = "box_img";                                 // div 类名为box_img
                ccontent.appendChild(boximg);                                 // 父级上追加
                var img = document.createElement("img");            // 最后增加个img标签
                img.src = imgData.data[i].src;                                // 指定src属性
                boximg.appendChild(img);                                      // 父级上追加
            }
            imgLocation("container", 'box');                  // 重新执行排列一下
        }
    }
}

function checkFlag() {
    // true运行加载数据
    var cparent = document.getElementById("container");                           // 获取父级空间
    var ccontent = getChildElement(cparent,"box");
    var lastContentHeight = ccontent[ccontent.length - 1].offsetTop;                        // 获取最后一张图片距顶部的高度
    var scrollTop = document.documentElement.scrollTop || document.body.scrollTop;          // 滚动条当前位置高度(document.body.scrollTop是避免浏览器兼容问题)
    var pageHeight = document.documentElement.clientHeight || document.body.clientHeight;   // 当前页面高度
    // console.log(lastContentHeight+':'+scrollTop+":"+pageHeight);
    if(lastContentHeight<scrollTop+pageHeight){                                             // 判断是否到底部
        return true;
    }
}


function imgLocation(parent, content) {
    // 将父级空间parent下的所有子级空间content中的内容全部取出
    var cparent = document.getElementById(parent);                              // 获取父级空间
    var ccontent = getChildElement(cparent, content);                           // 获取子级空间里的内容
    var imgWidth = ccontent[0].offsetWidth;                                     // 获取图片的宽度(每张图片都一样宽)
    var cols = Math.floor(document.documentElement.clientWidth / imgWidth);  // 获取一行的图片数=屏幕宽度/图片宽度
    cparent.style.cssText = "width:"+imgWidth*cols+"px;margin:0 auto";          // 设置父级空间的css宽度和边距(居中)

    var boxHeight = [];                                                         // 存储第一行所有盒子高度
    for(var i=0;i<ccontent.length;i++){                                         // 循环盒子数
        if(i<cols){                                                             // 如果i小于一行的图片数(就是第一行的个数)
            boxHeight[i] = ccontent[i].offsetHeight;                            // 获取每张图片高度
        }else {
            var minHeight = Math.min.apply(null, boxHeight);            // 获取数组中最小高度
            var minIndex = getMinHeightLocation(boxHeight, minHeight);          //获取最小高度的位置的索引值
            // console.log(minHeight);
            ccontent[i].style.position = "absolute";                            // 设置定位为绝对布局
            ccontent[i].style.top = minHeight+"px";                             // 设置高度
            ccontent[i].style.left = ccontent[minIndex].offsetLeft+"px";        // 根据最小高度设置居左宽度
            boxHeight[minIndex] = boxHeight[minIndex]+ccontent[i].offsetHeight; // 填充后高度就不是最小了，为两个的高度
        }
    }
}

function getChildElement(parent,content) {
    var contentArr = [];  //存储子级空间的内容
    var allContent = parent.getElementsByTagName("*");  //使用通配符“*”
    for(var i=0;i<allContent.length;i++){
        if(allContent[i].className == content){         // 如果子级空间里内容的类名为content时存储起来
            contentArr.push(allContent[i]);             // 末尾追加
        }
    }
    return contentArr;
}

function getMinHeightLocation(boxHeightArr,minHeight) {
    // 获取最小高度的位置
    for(var i in boxHeightArr){
        if(boxHeightArr[i] == minHeight)                // 如果高度等于最小高度
            return i;
    }
}