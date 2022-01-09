# Python编码规范指南
详情参考：https://www.python.org/dev/peps/pep-0008/

## 缩进

        每级缩进使用4个空格。
        
        连续行应该对齐折叠元素，无论是垂直的Python的隐式行连接圆括号内的，中括号内的，大括号内的，还是使用悬挂缩进[5]。
        
        使用悬挂缩进应注意以下几点；
        
            1、第一行没有参数并且使用更多的缩进来区别它本身和连续行。
                风格良好：  
                
                    # 与分界符对齐。
                    foo = long_function_name(var_one, var_two,
                                             var_three, var_four)
                    
                    # 包括更多的缩进以区别于其他的。
                    def long_function_name(
                            var_one, var_two, var_three,
                            var_four):
                        print(var_one)
                    
                    # 悬挂缩进应增加一个级别
                    foo = long_function_name(
                        var_one, var_two,
                        var_three, var_four)
                风格不良：   
                    
                    # 第一行参数禁止不使用垂直对齐
                    foo = long_function_name(var_one, var_two,
                        var_three, var_four)
                    
                    # 当无法区分缩进时，需要进一步缩进
                    def long_function_name(
                        var_one, var_two, var_three,
                        var_four):
                        print(var_one)
            
            2、对于连续行，4个空格规则是可选的。
                
                可选的：
                    
                    # 悬挂缩进可能缩进不是4个空格
                    foo = long_function_name(
                      var_one, var_two,
                      var_three, var_four)
    
            3、if语句条件块足够长时需要编写多行，值得注意的是两个字符组成的关键字（例如if），加上一个空格，加上开括号为多行条件的后续行创建一个4个空格的缩进。
            这可以给嵌入if内的缩进语句产生视觉冲突，这也自然被缩进4个空格。这个PEP没有明确如何（是否）进一步区分条件行和if语句内的嵌入行。
            这种情况下，可以接受的选项包括，但不仅限于：
                
                # 没有额外的缩进
                if (this_is_one_thing and
                    that_is_another_thing):
                    do_something()
                
                # 添加一行注释，这将为编辑器支持语法高亮提供一些区分。
                # supporting syntax highlighting.
                if (this_is_one_thing and
                    that_is_another_thing):
                    # Since both conditions are true, we can frobnicate.
                    do_something()
                
                # 在条件连接行，增加额外的缩进
                if (this_is_one_thing
                        and that_is_another_thing):
                    do_something()
                    
            4、多行结构中的结束花括号/中括号/圆括号是最后一行的第一个非空白字符，
            如：
                my_list = [
                    1, 2, 3,
                    4, 5, 6,
                    ]
                result = some_function_that_takes_arguments(
                    'a', 'b', 'c',
                    'd', 'e', 'f',
                    )
            或者是最后一行的第一个字符，
            如：
                my_list = [
                    1, 2, 3,
                    4, 5, 6,
                ]
                result = some_function_that_takes_arguments(
                    'a', 'b', 'c',
                    'd', 'e', 'f',
                )
            
## 制表符还是空格？

        空格是缩进方法的首选。
        制表符仅用于与已经用制表符做缩进的代码保持一致。
        Python3不允许混用制表符和空格来缩进。
        Python2代码混用制表符和空格缩进，将被转化为只使用空格。
        调用Python2命令行解释器时使用-t选项，可对代码中非法混用制表符和空格发出警告。当使用-tt选项，警告将变成错误。这些选项是高度推荐的！

## 行的最大长度
        
        限制所有行最多79个字符。

        下垂的长块结构限制为更少的文本（文档字符串或注释），行的长度应该限制在72个字符。
        
        限制编辑器窗口宽度使得并排打开多个文件成为可能，并且使用代码审查工具显示相邻列的两个版本工作正常。
        
        绝大多数工具的默认折叠会破坏代码的可视化结构，使其更难以理解。编辑器中的窗口宽度设置为80个字符。即使该工具将在最后一列中标记
        字形。一些基于网络的工具可能不会提供动态的自动换行。
        
        有些团队强烈喜欢较长的行长度。对于代码维护完全或主要由一个团队的，可以在这个问题上达成协议，象征性的将行长度从80个字符增加到
        100个字符（有效地增加最大长度到99个字符）也是可以的，提供注释和文档字符串仍是72个字符。
        
        Python标准库采取保守做法，要求行限制到79个字符（文档字符串/注释到72个字符）。
        
        折叠长行的首选方法是在小括号，中括号，大括号中使用Python隐式换行。长行可以在表达式外面使用小括号来变成多行。连续行使用反斜杠更好。
        
        反斜杠有时可能仍然是合适的。例如，长的多行的with语句不能用隐式续行，可以用反斜杠：
        
            with open('/path/to/some/file/you/want/to/read') as file_1, \
                 open('/path/to/some/file/being/written', 'w') as file_2:
                file_2.write(file_1.read())

            （为进一步思考With语句的多行缩进，见前面多行if语句的讨论。）
            
            另一个这样的例子是assert语句。
            
            确保适当的连续行缩进。
            
## 换行应该在二元操作符的前面还是后面？
    
        风格良好：
        
            # 好的做法：很容易看出二元操作符和被操作对象的关系
            income = (gross_wages
              + taxable_interest
              + (dividends - qualified_dividends)
              - ira_deduction
              - student_loan_interest)
              
        风格不良：
        
            # 不好的做法：操作符和被操作符的对象是分离的
            income = (gross_wages +
              taxable_interest +
              (dividends - qualified_dividends) -
              ira_deduction -
              student_loan_interest)

## 空行

        顶级函数和类的定义之间有两行空行。

        类内部的函数定义之间有一行空行。
        
        额外的空行用来（谨慎地）分离相关的功能组。相关的行（例如：一组虚拟实现）之间不使用空行。
        
        在函数中谨慎地使用空行来表示逻辑部分。
        
        Python接受control-L（即^L）换页符作为空白符；许多工具把这些字符作为分页符，所以你可以使用它们为文件中的相关部分分页。
        注意，一些编辑器和基于Web的代码查看器可能不能识别control-L是换页，将显示另外的字形。
        
## 源文件编码
        
        在源文件中一直使用utf-8编码，在python2中使用ascll编码。

        文件，在python2 中使用ascll编码，在python3中使用utf-8编码

## 导入
        
        导入通常是单独一行，例如：
        风格良好：
        
            import os
            import sys
            from subprocess import Popen, PIPE
        
        风格不良：
        
            import sys, os    
            
        导入常常位于文件顶部，在模块注释和字符串文档之后，在模块的全局变量和常量之前。

        导入应该按照以下顺序分组：
            1. 标准库导入
            2. 相关的第三方导入
            3. 特定的本地应用/库导入
        在每个导入组之间放一行空行。
        
        把任何相关__all__规范放在导入之后。
        
        推荐绝对导入，因为它们更易读，并且如果导入系统配置的不正确（例如当包中的一个目录结束于sys.path）
        它们有更好的表现（至少给出更好的错误信息）：
        
            import mypkg.sibling
            from mypkg import sibling
            from mypkg.sibling import example
            
        明确的相对导入可以用来接受替代绝对导入，特别是处理复杂包布局时，绝对导入过于冗长。
            
            from . import sibling
            from .sibling import example
            
        标准库代码应该避免复杂包布局并使用绝对导入。
        
        隐式的相对导入应该永远不被使用，并且在Python3中已经移除。
        
        从一个包含类的模块中导入类时，通常下面这样是好的写法：
        
            from myclass import MyClass
            from foo.bar.yourclass import YourClass
            
        如果这种写法导致本地名字冲突，那么就这样写：
            
            import myclass
            import foo.bar.yourclass
            
        并使用“myclass.MyClass”和“foo.bar.yourclass.YourClass”来访问。

        避免使用通配符导入（from <模块名> import *），因为它们使哪些名字出现在命名空间变得不清楚，这混淆了读者和许多自动化工具。
        通配符导入有一种合理的使用情况，重新发布一个内部接口作为一个公共API的一部分（例如，重写一个纯Python实现的接口，
        该接口定义从一个可选的加速器模块并且哪些定义将被重写提前并不知道）。
        
        用这种方式重新命名，下面的有关公共和内部接口的指南仍适用。
        
## 模块级别的内置属性
        
        模块级别的内置属性（名字有前后双下划线的），例如__all__, __author__, __version__，应该放置在模块的文档字符串后，
        任意import语句之前，from __future__导入除外。Python强制要求from __future__导入必须在任何代码之前，只能在模块级文档字符串之后。
            
            """This is the example module.

            This module does stuff.
            """
            
            from __future__ import barry_as_FLUFL
            
            __all__ = ['a', 'b', 'c']
            __version__ = '0.1'
            __author__ = 'Cardinal Biggles'
            
            import os
            import sys

## 字符串引号

        Python中，单引号字符串和双引号字符串是一样的。本PEP不建议如此。建议选择一条规则并坚持下去。当一个字符串包含单引号字符或双引号字符时，使用另一种字符串引号来避免字符串中使用反斜杠。这提高可读性。

        三引号字符串，与PEP 257 文档字符串规范一致总是使用双引号字符。
    
## 表达式和语句中的空格
    
        以下情况避免使用多余的空格：
             
        紧挨着小括号，中括号或大括号。
        
            Yes: spam(ham[1], {eggs: 2})
            No:  spam( ham[ 1 ], { eggs: 2 } )
        
        紧挨在逗号，分号或冒号前：
        
            Yes: if x == 4: print x, y; x, y = y, x
            No:  if x == 4 : print x , y ; x , y = y , x

        在切片中冒号像一个二元操作符，冒号两侧的有相等数量空格（把它看作最低优先级的操作符）。在一个扩展切片中，两个冒号必须有相等数量的空格。
        例外：当一个切片参数被省略时，该空格被省略。
            
            风格良好：
            
                ham[1:9], ham[1:9:3], ham[:9:3], ham[1::3], ham[1:9:]
                ham[lower:upper], ham[lower:upper:], ham[lower::step]
                ham[lower+offset : upper+offset]
                ham[: upper_fn(x) : step_fn(x)], ham[:: step_fn(x)]
                ham[lower + offset : upper + offset]
                
            风格不良：
                
                ham[lower + offset:upper + offset]
                ham[1: 9], ham[1 :9], ham[1:9 :3]
                ham[lower : : upper]
                ham[ : upper]

        紧挨着左括号之前，函数调用的参数列表的开始处：
        
            Yes: spam(1)
            No:  spam (1)
            
        紧挨着索引或切片开始的左括号之前：
        
            Yes: dct['key'] = lst[index]
            No:  dct ['key'] = lst [index]
        
        为了与另外的赋值（或其它）操作符对齐，不止一个空格。
            
            Yes:
            
                x = 1
                y = 2
                long_variable = 3
                
            No:
            
                x             = 1
                y             = 2
                long_variable = 3
        
## 其它建议

            始终避免行尾空白。因为它们通常不可见，容易导致困惑：如果\后面跟了一个空格，它就不是一个有效的续行符了。
            很多编辑器不保存行尾空白，CPython项目中也设置了commit前检查以拒绝行尾空白的存在。
    
            始终在这些二元操作符的两边放置一个空格：赋值（= ），增强赋值（+= ，-= 等），
            比较（== ， < ， > ， != ， <> ， <= ， >= ，in ， not in ，is ，is not ），布尔（and ，or ，not ）。
            
            如果使用了不同优先级的操作符，在低优先级操作符周围增加空格（一个或多个）。不要使用多于一个空格，二元运算符两侧空格数量相等。
        
                Yes:

                    i = i + 1
                    submitted += 1
                    x = x*2 - 1
                    hypot2 = x*x + y*y
                    c = (a+b) * (a-b)
                
                No:
                
                    i=i+1
                    submitted +=1
                    x = x * 2 - 1
                    hypot2 = x * x + y * y
                    c = (a + b) * (a - b)

            当=符号用于指示关键字参数或默认参数值时，它周围不要使用空格。
            
                Yes:

                    def complex(real, imag=0.0):
                        return magic(r=real, i=imag)
                    
                No:
                
                    def complex(real, imag = 0.0):
                        return magic(r = real, i = imag)
                        
            带注解的函数使用正常的冒号规则，并且在->两侧增加一个空格：
            
                Yes:

                    def munge(input: AnyStr): ...
                    def munge() -> AnyStr: ...
                    
                No:
                
                    def munge(input:AnyStr): ...
                    def munge()->PosInt: ...

            如果参数既有注释又有默认值，在等号两边增加一个空格（仅在既有注释又有默认值时才加这个空格）。
                
                Yes:
                
                    def munge(sep: AnyStr = None): ...
                    def munge(input: AnyStr, sep: AnyStr = None, limit=1000): ...
                
                No:
                
                    def munge(input: AnyStr=None): ...
                    def munge(input: AnyStr, limit = 1000): ...

            不鼓励使用复合语句（同一行有多条语句）。
                
                风格良好:

                    if foo == 'blah':
                        do_blah_thing()
                    do_one()
                    do_two()
                    do_three()
                
                最好不要:
                
                    if foo == 'blah': do_blah_thing()
                    do_one(); do_two(); do_three()

            尽管有时if/for/while的同一行跟一小段代码，在一个多条子句的语句中不要如此。避免折叠长行！
                
                最好不要:

                    if foo == 'blah': do_blah_thing()
                    for x in lst: total += x
                    while t < 10: t = delay()
                
                绝对不要:
                    
                    if foo == 'blah': do_blah_thing()
                    else: do_non_blah_thing()
                    
                    try: something()
                    finally: cleanup()
                    
                    do_one(); do_two(); do_three(long, argument,
                                                 list, like, this)
                    
                    if foo == 'blah': one(); two(); three()
            
## 什么时候使用尾部逗号？
    
        尾部逗号通常都是可选的，除了一些强制的场景，比如元组在只有一个元素的时候需要一个尾部逗号。
        为了代码更加清晰，元组只有一个元素时请务必用括号括起来（语法上没有强制要求）：
            
            Yes:
    
                FILES = ('setup.cfg',)
            
            OK, but confusing:
            
                FILES = 'setup.cfg',
        
        当尾部逗号不是必须时，如果你用了版本控制系统那么它将很有用。当列表元素、参数、导入项未来可能不断增加时，留一个尾部逗号是一个很好的选择。
        通常的用法是（比如列表）每个元素独占一行，然后尾部都有逗号，在最后一个元素的下一行写闭标签。如果你的数据结构都是写在同一行的，就没有必要保留尾部逗号了。
    
            Yes:
    
                FILES = [
                    'setup.cfg',
                    'tox.ini',
                    ]
                initialize(FILES,
                           error=True,
                           )
            No:
                
                FILES = ['setup.cfg', 'tox.ini',]
                initialize(FILES, error=True,)

## 注释

        同代码相矛盾的注释比没有注释更差。当代码修改时，始终优先更新注释！
    
        注释应该是完整的句子。如果注释是一个短语或句子，它的第一个单词的首字母应该大写，除非它是一个以小写字母开头的标识符（不更改标识符的情况下！）。
        
        如果注释很短，末尾可以不加句号。注释块通常由一个或多个段落组成，这些段落由完整的句子组成，并且每个句子都应该以句号结尾。
        
        在句尾的句号后边使用两个空格。
        
        写英语注释时，遵循断词和空格。
        
        非英语国家的Python程序员：请用英语书写注释，除非你120%的确定，所有看你代码的人都和你说一样的语言。
        
        非英语国家的Python程序员：请写下你的意见，在英语中，除非你是120%肯定，代码将不会被不讲你的语言的人阅读。

## 注释块

        注释块通常适用于一些（或全部）紧跟其后的代码，并且那些代码应使用相同级别的缩进。注释块的每行以一个#和一个空格开始（除非注释里面的文本有缩进）。
        
        注释块内的段落之间由仅包含#的行隔开。

## 行内注释

        谨慎地使用行内注释。
        
        行内注释就是注释和代码在同一行，它与代码之间至少用两个空格隔开。并且它以#和一个空格开始。
        
        如果行内注释指出的是显而易见，那么它就是不必要的。不要使用无效注释，主要是说明其目的
        
            不要这样做：
        
                x = x + 1                 # Increment x
            
            But sometimes, this is useful:
            
                x = x + 1                 # Compensate for border


## 文档字符串

        编写好的文档字符串（即“代码”）约定在PEP 257中是永存的。
        
        为所有公共模块，函数，类和方法书写文档字符串。对非公开的方法书写文档字符串是没有必要的，但应该写注释描述这个方法是做什么的。
        这些注释应该写在def行后面。
        
        PEP 257描述了好的文档字符串约定。最重要的是，多行文档字符串以一行"""结束，例如：

        yes:
        
            """Return a foobang

            Optional plotz says to frobnicate the bizbaz first.
            """
            
        对于只有一行的文档字符串，"""同一行上。

## 命名规范

        使用单独的小写字母（b）
        
        使用单独的大写字母（B）
        
        使用小写字母（lowercase）
        
        使用小写字母和下划线（lower_case_with_underscores）
        
        使用大写字母（UPPERCASE）
        
        使用大写字母和下划线（UPPER_CASE_WITH_UPPERCASE）
        
        驼峰式写法（CamelCase）：在使用缩写的时候，大写优于小写例如HTTPServer优于HttpServer
        
        首字母大写，然后使用下划线是一种丑陋的写法
        
        1、避免使用的名称
        
            在写变量的时候，尽量避免小写的l和大写字母O和大写字母I，主要原因是容易和数字中1,0相混淆。当想使用‘l’时，用‘L’代替。

        2、包名和模块名
            
            模块尽量使用简短的全部小写的名称，如果可以增加可读性那么可以使用下划线，python的包不推荐使用下划线，
            但是在引用其他语言写的扩展包中可以使用下划线来表示区分
            
        3、类名称
        
            类名称主要遵循为CapWords约定，表示为首字母大写
        
        4、类型变量名称

            类型变量名称应该首字母大写，并且尽量短，比如：T, AnyStr, Num。对于协变量和有协变行为的变量，建议添加后缀__co或者__contra。

        5、异常名
            
            因为异常应该是类，所以类的命名规则在这里也同样适用。然而，异常名（如果这个异常确实是一个错误）应该使用后缀“Error”。

        6、全局变量名

            （希望这些变量是在一个模块内使用。）这些规则和那些有关函数的规则是相同的。
            
            模块设计为通过from M import *来使用，应使用__all__机制防止导出全局变量，或使用加前缀的旧规则，为全局变量加下划线（可能你像表明这些全局变量是“非公开模块”）。
            
        7、函数名

            函数名应该是小写字母，必要时单词用下划线分开以提高可读性。
                        
            混合大小写仅用于这种风格已经占主导地位的上下文（例如threading.py），以保持向后兼容性。

        8、函数和方法参数

            使用self做实例化方法的第一个参数。
            
            使用cls做类方法的第一个参数。     
            
            如果函数的参数名与保留关键字冲突，最好是为参数名添加一个后置下划线而不是使用缩写或拼写错误。
          
            因此class_ 比clss好。（也许使用同义词来避免更好。）。
        
        9、常量
        
            常量通常定义于模块级别并且所有的字母都是大写，单词用下划线分开。例如MAX_OVERFLOW和TOTAL。

















