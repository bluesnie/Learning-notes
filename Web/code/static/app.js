// 对比
$(document).ready(function () {
   $("#a-btn").button()
});


// 拖动
$(document).ready(function () {
    $("#div1").draggable();
    // 放置
    $("#div2").droppable();

    // 放置事件
    $("#div2").on("drop", function (event,ui) {
        // alert(event);
        $("#div2").text("drop事件")
    })
});

// resizeable
$(document).ready(function () {
   $("#div3").resizable();
});

// 选择题
$(document).ready(function () {
   $("#btn").button();
   $("#select").selectable();
   $("#btn").on("click", function () {
       if($("#right").hasClass("ui-selected")){
           alert("yes")
       }
   });
});

// 拖动排序
$(document).ready(function () {
   $("#sortable").sortable();
});

// 折叠选项卡
$(document).ready(function () {
   $("#accordion").accordion();
});

// 自动提示
$(document).ready(function () {
   var autoTags = ['jack','ime','html','css','python','java','php']
    $('#tags').autocomplete({
       source:autoTags
    });
});

// 日期选择
$(document).ready(function () {
   $("#datepicker").datepicker();
   $("#date-btn").button();
});

// 对话框
$(document).ready(function () {
   $("#dialog-btn").button().on('click', function () {
      $("#dialog-div").dialog();
   })
});

// 进度条
var pb;
var max=100;
var current=0;

$(document).ready(function () {
   pb = $("#pb");
   pb.progressbar({max:100});
   setInterval(changepb, 100)

});

function changepb() {
    current++;
    if(current>=100){
        current=0;
    }
    pb.progressbar("option",'value', current);
}