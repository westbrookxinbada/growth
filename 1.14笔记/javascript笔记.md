

##### 1如何引入在页面标签的图像



```javascript

<link rel="shortcut icon" href="img/favicon.ico" type="image/x-icon">
```

##### 2如何页面跳转



```
var time = document.getElementById("time");
var second = 5;

function showTime() {
    second -- ;
    if (second<=0){
        location.href = "recommend1.html";
    }
    time.innerHTML = second;
}
setInterval(showTime,1000);
```

```

```

##### 3如何更换图像

```
//修改图片src属性
var number = 1;
function fun(){
    number ++ ;
    //判断number是否大于3
    if(number > 3){
        number = 1;
    }
    //获取img对象
    var img = document.getElementById("img");
    img.src = "img/banner_"+number+".jpg";
}

//2.定义定时器
setInterval(fun,3000);
```