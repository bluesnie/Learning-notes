###### datetime:2022/11/05 15:42

###### author:nzb

# 5 STL- 常用算法

概述

* 算法主要是由头文件`<algorithm>` `<functional>` `<numeric>`组成。
* `<algorithm>`是所有STL头文件中最大的一个，范围涉及到比较、 交换、查找、遍历操作、复制、修改等等
* `<numeric>`体积很小，只包括几个在序列上面进行简单数学运算的模板函数
* `<functional>`定义了一些模板类,用以声明函数对象。

### 5.1 常用遍历算法

学习目标：掌握常用的遍历算法

算法简介

* `for_each`     //遍历容器
* `transform`   //搬运容器到另一个容器中

#### 5.1.1 for_each

功能描述：实现遍历容器

函数原型

* `for_each(iterator beg, iterator end, _func);  `

    - 遍历算法 遍历容器元素
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `_func` 函数或者函数对象

示例

```C++
#include <algorithm>
#include <vector>

//普通函数
void print01(int val) 
{
	cout << val << " ";
}
//函数对象
class print02 
{
 public:
	void operator()(int val) 
	{
		cout << val << " ";
	}
};

//for_each算法基本用法
void test01() {

	vector<int> v;
	for (int i = 0; i < 10; i++) 
	{
		v.push_back(i);
	}

	//遍历算法
	for_each(v.begin(), v.end(), print01);
	cout << endl;

	for_each(v.begin(), v.end(), print02());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：for_each在实际开发中是最常用遍历算法，需要熟练掌握

#### 5.1.2 transform

功能描述：搬运容器到另一个容器中

函数原型

* `transform(iterator beg1, iterator end1, iterator beg2, _func);`
    - `beg1` 源容器开始迭代器
    - `end1` 源容器结束迭代器
    - `beg2` 目标容器开始迭代器
    - `_func` 函数或者函数对象

示例

```C++
#include<vector>
#include<algorithm>

//常用遍历算法  搬运 transform

class TransForm
{
public:
	int operator()(int val)
	{
		return val;
	}

};

class MyPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int>v;
	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}

	vector<int>vTarget; //目标容器

	vTarget.resize(v.size()); // 目标容器需要提前开辟空间

	transform(v.begin(), v.end(), vTarget.begin(), TransForm());

	for_each(vTarget.begin(), vTarget.end(), MyPrint());
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：搬运的目标容器必须要提前开辟空间，否则无法正常搬运

### 5.2 常用查找算法

学习目标：掌握常用的查找算法

算法简介

- `find`                     //查找元素
- `find_if`               //按条件查找元素
- `adjacent_find`    //查找相邻重复元素
- `binary_search`    //二分查找法
- `count`                   //统计元素个数
- `count_if`             //按条件统计元素个数

#### 5.2.1 find

功能描述：查找指定元素，找到返回指定元素的迭代器，找不到返回结束迭代器`end()`

函数原型

- `find(iterator beg, iterator end, value);  `
    - 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `value` 查找的元素

示例

```C++
#include <algorithm>
#include <vector>
#include <string>
void test01() {

	vector<int> v;
	for (int i = 0; i < 10; i++) {
		v.push_back(i + 1);
	}
	//查找容器中是否有 5 这个元素
	vector<int>::iterator it = find(v.begin(), v.end(), 5);
	if (it == v.end()) 
	{
		cout << "没有找到!" << endl;
	}
	else 
	{
		cout << "找到:" << *it << endl;
	}
}

class Person {
public:
	Person(string name, int age) 
	{
		this->m_Name = name;
		this->m_Age = age;
	}
	//重载==
	bool operator==(const Person& p) 
	{
		if (this->m_Name == p.m_Name && this->m_Age == p.m_Age) 
		{
			return true;
		}
		return false;
	}

public:
	string m_Name;
	int m_Age;
};

void test02() {

	vector<Person> v;

	//创建数据
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);

	vector<Person>::iterator it = find(v.begin(), v.end(), p2);
	if (it == v.end()) 
	{
		cout << "没有找到!" << endl;
	}
	else 
	{
		cout << "找到姓名:" << it->m_Name << " 年龄: " << it->m_Age << endl;
	}
}
```

总结： 利用find可以在容器中找指定的元素，返回值是**迭代器**

#### 5.2.2 find_if

功能描述：按条件查找元素

函数原型

- `find_if(iterator beg, iterator end, _Pred);  `
    - 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `_Pred` 函数或者谓词（返回bool类型的仿函数）

示例

```C++
#include <algorithm>
#include <vector>
#include <string>

//内置数据类型
class GreaterFive
{
public:
	bool operator()(int val)
	{
		return val > 5;
	}
};

void test01() {

	vector<int> v;
	for (int i = 0; i < 10; i++) {
		v.push_back(i + 1);
	}

	vector<int>::iterator it = find_if(v.begin(), v.end(), GreaterFive());
	if (it == v.end()) {
		cout << "没有找到!" << endl;
	}
	else {
		cout << "找到大于5的数字:" << *it << endl;
	}
}

//自定义数据类型
class Person {
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}
public:
	string m_Name;
	int m_Age;
};

class Greater20
{
public:
	bool operator()(Person &p)
	{
		return p.m_Age > 20;
	}

};

void test02() {

	vector<Person> v;

	//创建数据
	Person p1("aaa", 10);
	Person p2("bbb", 20);
	Person p3("ccc", 30);
	Person p4("ddd", 40);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);

	vector<Person>::iterator it = find_if(v.begin(), v.end(), Greater20());
	if (it == v.end())
	{
		cout << "没有找到!" << endl;
	}
	else
	{
		cout << "找到姓名:" << it->m_Name << " 年龄: " << it->m_Age << endl;
	}
}

int main() {

	//test01();

	test02();

	system("pause");

	return 0;
}
```

总结：find_if按条件查找使查找更加灵活，提供的仿函数可以改变不同的策略

#### 5.2.3 adjacent_find

功能描述：查找相邻重复元素

函数原型

- `adjacent_find(iterator beg, iterator end);  `
    - 查找相邻重复元素,返回相邻元素的第一个位置的迭代器
    - `beg` 开始迭代器
    - `end` 结束迭代器

示例

```C++
#include <algorithm>
#include <vector>

void test01()
{
	vector<int> v;
	v.push_back(1);
	v.push_back(2);
	v.push_back(5);
	v.push_back(2);
	v.push_back(4);
	v.push_back(4);
	v.push_back(3);

	//查找相邻重复元素
	vector<int>::iterator it = adjacent_find(v.begin(), v.end());
	if (it == v.end()) {
		cout << "找不到!" << endl;
	}
	else {
		cout << "找到相邻重复元素为:" << *it << endl;
	}
}
```

总结：面试题中如果出现查找相邻重复元素，记得用STL中的adjacent_find算法

#### 5.2.4 binary_search

功能描述：查找指定元素是否存在

函数原型

- `bool binary_search(iterator beg, iterator end, value);  `
    - 查找指定的元素，查到 返回true 否则false
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `value` 查找的元素

- 注意: 在**无序序列中不可用**

示例

```C++
#include <algorithm>
#include <vector>

void test01()
{
	vector<int>v;

	for (int i = 0; i < 10; i++)
	{
		v.push_back(i);
	}
	//二分查找
	bool ret = binary_search(v.begin(), v.end(),2);
	if (ret)
	{
		cout << "找到了" << endl;
	}
	else
	{
		cout << "未找到" << endl;
	}
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**二分查找法查找效率很高，值得注意的是查找的容器中元素必须的有序序列

#### 5.2.5 count

功能描述：统计元素个数

函数原型

- `count(iterator beg, iterator end, value);  `
    - 统计元素出现次数
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `value` 统计的元素

示例

```C++
#include <algorithm>
#include <vector>

//内置数据类型
void test01()
{
	vector<int> v;
	v.push_back(1);
	v.push_back(2);
	v.push_back(4);
	v.push_back(5);
	v.push_back(3);
	v.push_back(4);
	v.push_back(4);

	int num = count(v.begin(), v.end(), 4);

	cout << "4的个数为： " << num << endl;
}

//自定义数据类型
class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}
	bool operator==(const Person & p)
	{
		if (this->m_Age == p.m_Age)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	string m_Name;
	int m_Age;
};

void test02()
{
	vector<Person> v;

	Person p1("刘备", 35);
	Person p2("关羽", 35);
	Person p3("张飞", 35);
	Person p4("赵云", 30);
	Person p5("曹操", 25);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);
    
    Person p("诸葛亮",35);

	int num = count(v.begin(), v.end(), p);
	cout << "num = " << num << endl;
}
int main() {

	//test01();

	test02();

	system("pause");

	return 0;
}
```

总结： 统计自定义数据类型时候，需要配合重载 `operator==`

#### 5.2.6 count_if

功能描述：按条件统计元素个数

函数原型

- `count_if(iterator beg, iterator end, _Pred);  `
    - 按条件统计元素出现次数
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `_Pred` 谓词

示例

```C++
#include <algorithm>
#include <vector>

class Greater4
{
public:
	bool operator()(int val)
	{
		return val >= 4;
	}
};

//内置数据类型
void test01()
{
	vector<int> v;
	v.push_back(1);
	v.push_back(2);
	v.push_back(4);
	v.push_back(5);
	v.push_back(3);
	v.push_back(4);
	v.push_back(4);

	int num = count_if(v.begin(), v.end(), Greater4());

	cout << "大于4的个数为： " << num << endl;
}

//自定义数据类型
class Person
{
public:
	Person(string name, int age)
	{
		this->m_Name = name;
		this->m_Age = age;
	}

	string m_Name;
	int m_Age;
};

class AgeLess35
{
public:
	bool operator()(const Person &p)
	{
		return p.m_Age < 35;
	}
};
void test02()
{
	vector<Person> v;

	Person p1("刘备", 35);
	Person p2("关羽", 35);
	Person p3("张飞", 35);
	Person p4("赵云", 30);
	Person p5("曹操", 25);

	v.push_back(p1);
	v.push_back(p2);
	v.push_back(p3);
	v.push_back(p4);
	v.push_back(p5);

	int num = count_if(v.begin(), v.end(), AgeLess35());
	cout << "小于35岁的个数：" << num << endl;
}


int main() {

	//test01();

	test02();

	system("pause");

	return 0;
}
```

总结：按值统计用count，按条件统计用count_if

### 5.3 常用排序算法

学习目标：掌握常用的排序算法

算法简介

- `sort`             //对容器内元素进行排序
- `random_shuffle`   //洗牌 指定范围内的元素随机调整次序
- `merge `           // 容器元素合并，并存储到另一容器中
- `reverse`       // 反转指定范围的元素

#### 5.3.1 sort

功能描述：对容器内元素进行排序

函数原型

- `sort(iterator beg, iterator end, _Pred);  `
    - 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `_Pred` 谓词

示例

```c++
#include <algorithm>
#include <vector>

void myPrint(int val)
{
	cout << val << " ";
}

void test01() {
	vector<int> v;
	v.push_back(10);
	v.push_back(30);
	v.push_back(50);
	v.push_back(20);
	v.push_back(40);

	//sort默认从小到大排序
	sort(v.begin(), v.end());
	for_each(v.begin(), v.end(), myPrint);
	cout << endl;

	//从大到小排序
	sort(v.begin(), v.end(), greater<int>());
	for_each(v.begin(), v.end(), myPrint);
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：sort属于开发中最常用的算法之一，需熟练掌握

#### 5.3.2 random_shuffle

功能描述：洗牌 指定范围内的元素随机调整次序

函数原型

- `random_shuffle(iterator beg, iterator end);  `
    - 指定范围内的元素随机调整次序
    - `beg` 开始迭代器
    - `end` 结束迭代器

示例

```c++
#include <algorithm>
#include <vector>
#include <ctime>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	srand((unsigned int)time(NULL));
	vector<int> v;
	for(int i = 0 ; i < 10;i++)
	{
		v.push_back(i);
	}
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;

	//打乱顺序
	random_shuffle(v.begin(), v.end());
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：random_shuffle洗牌算法比较实用，**使用时记得加随机数种子**

#### 5.3.3 merge

功能描述：两个容器元素合并，并存储到另一容器中

函数原型

- `merge(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);  `
    - 容器元素合并，并存储到另一容器中
    - `beg1` 容器1开始迭代器
    - `end1` 容器1结束迭代器
    - `beg2` 容器2开始迭代器
    - `end2` 容器2结束迭代器
    - `dest` 目标容器开始迭代器

- 注意: 两个容器必须是**有序的**

示例

```c++
#include <algorithm>
#include <vector>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v1;
	vector<int> v2;
	for (int i = 0; i < 10 ; i++) 
    {
		v1.push_back(i);
		v2.push_back(i + 1);
	}

	vector<int> vtarget;
	//目标容器需要提前开辟空间
	vtarget.resize(v1.size() + v2.size());
	//合并  需要两个有序序列
	merge(v1.begin(), v1.end(), v2.begin(), v2.end(), vtarget.begin());
	for_each(vtarget.begin(), vtarget.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：merge合并的两个容器必须的有序序列

#### 5.3.4 reverse

功能描述：将容器内元素进行反转

函数原型

- `reverse(iterator beg, iterator end);  `
    - 反转指定范围的元素
    - `beg` 开始迭代器
    - `end` 结束迭代器

示例

```c++
#include <algorithm>
#include <vector>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v;
	v.push_back(10);
	v.push_back(30);
	v.push_back(50);
	v.push_back(20);
	v.push_back(40);

	cout << "反转前： " << endl;
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;

	cout << "反转后： " << endl;

	reverse(v.begin(), v.end());
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**reverse反转区间内元素，面试题可能涉及到

### 5.4 常用拷贝和替换算法

学习目标：掌握常用的拷贝和替换算法

算法简介

- `copy`                      // 容器内指定范围的元素拷贝到另一容器中
- `replace`                // 将容器内指定范围的旧元素修改为新元素
- `replace_if `          // 容器内指定范围满足条件的元素替换为新元素
- `swap`                     // 互换两个容器的元素

#### 5.4.1 copy

功能描述：容器内指定范围的元素拷贝到另一容器中

函数原型

- `copy(iterator beg, iterator end, iterator dest);  `
    - 按值查找元素，找到返回指定位置迭代器，找不到返回结束迭代器位置
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `dest` 目标起始迭代器

示例

```c++
#include <algorithm>
#include <vector>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v1;
	for (int i = 0; i < 10; i++) {
		v1.push_back(i + 1);
	}
	vector<int> v2;
	v2.resize(v1.size());
	copy(v1.begin(), v1.end(), v2.begin());

	for_each(v2.begin(), v2.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：利用copy算法在拷贝时，目标容器记得提前开辟空间

#### 5.4.2 replace

功能描述：将容器内指定范围的旧元素修改为新元素

函数原型

- `replace(iterator beg, iterator end, oldvalue, newvalue);  `
    - 将区间内旧元素 替换成 新元素
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `oldvalue` 旧元素
    - `newvalue` 新元素

示例

```c++
#include <algorithm>
#include <vector>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v;
	v.push_back(20);
	v.push_back(30);
	v.push_back(20);
	v.push_back(40);
	v.push_back(50);
	v.push_back(10);
	v.push_back(20);

	cout << "替换前：" << endl;
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;

	//将容器中的20 替换成 2000
	cout << "替换后：" << endl;
	replace(v.begin(), v.end(), 20,2000);
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：replace会替换区间内满足条件的元素

#### 5.4.3 replace_if

功能描述: 将区间内满足条件的元素，替换成指定元素

函数原型

- `replace_if(iterator beg, iterator end, _pred, newvalue);  `
    - 按条件替换元素，满足条件的替换成指定元素
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `_pred` 谓词
    - `newvalue` 替换的新元素

示例

```c++
#include <algorithm>
#include <vector>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

class ReplaceGreater30
{
public:
	bool operator()(int val)
	{
		return val >= 30;
	}

};

void test01()
{
	vector<int> v;
	v.push_back(20);
	v.push_back(30);
	v.push_back(20);
	v.push_back(40);
	v.push_back(50);
	v.push_back(10);
	v.push_back(20);

	cout << "替换前：" << endl;
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;

	//将容器中大于等于的30 替换成 3000
	cout << "替换后：" << endl;
	replace_if(v.begin(), v.end(), ReplaceGreater30(), 3000);
	for_each(v.begin(), v.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：replace_if按条件查找，可以利用仿函数灵活筛选满足的条件

#### 5.4.4 swap

功能描述：互换两个容器的元素

函数原型

- `swap(container c1, container c2);  `
    - 互换两个容器的元素
    - c1容器1
    - c2容器2

示例

```c++
#include <algorithm>
#include <vector>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v1;
	vector<int> v2;
	for (int i = 0; i < 10; i++) {
		v1.push_back(i);
		v2.push_back(i+100);
	}

	cout << "交换前： " << endl;
	for_each(v1.begin(), v1.end(), myPrint());
	cout << endl;
	for_each(v2.begin(), v2.end(), myPrint());
	cout << endl;

	cout << "交换后： " << endl;
	swap(v1, v2);
	for_each(v1.begin(), v1.end(), myPrint());
	cout << endl;
	for_each(v2.begin(), v2.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

**总结：**swap交换容器时，注意交换的容器要同种类型

### 5.5 常用算术生成算法

学习目标：掌握常用的算术生成算法

注意

* 算术生成算法属于小型算法，使用时包含的头文件为 `#include <numeric>`

算法简介

- `accumulate`      // 计算容器元素累计总和
- `fill`                 // 向容器中添加元素

#### 5.5.1 accumulate

功能描述：计算区间内 容器元素累计总和

函数原型

- `accumulate(iterator beg, iterator end, value);  `
    - 计算容器元素累计总和
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `value` 起始值

示例

```c++
#include <numeric>
#include <vector>
void test01()
{
	vector<int> v;
	for (int i = 0; i <= 100; i++) {
		v.push_back(i);
	}

	int total = accumulate(v.begin(), v.end(), 0);

	cout << "total = " << total << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：accumulate使用时头文件注意是 numeric，这个算法很实用

#### 5.5.2 fill

功能描述：向容器中填充指定的元素

函数原型

- `fill(iterator beg, iterator end, value);  `
    - 向容器中填充元素
    - `beg` 开始迭代器
    - `end` 结束迭代器
    - `value` 填充的值

示例

```c++
#include <numeric>
#include <vector>
#include <algorithm>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{

	vector<int> v;
	v.resize(10);
	//填充
	fill(v.begin(), v.end(), 100);

	for_each(v.begin(), v.end(), myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结：利用fill可以将容器区间内元素填充为 指定的值

### 5.6 常用集合算法

学习目标：掌握常用的集合算法

算法简介

- `set_intersection`          // 求两个容器的交集
- `set_union`                       // 求两个容器的并集
- `set_difference `              // 求两个容器的差集

#### 5.6.1 set_intersection

功能描述：求两个容器的交集

函数原型

- `set_intersection(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);  `
    - 求两个集合的交集
    - `beg1` 容器1开始迭代器
    - `end1` 容器1结束迭代器
    - `beg2` 容器2开始迭代器
    - `end2` 容器2结束迭代器
    - `dest` 目标容器开始迭代器

- **注意:两个集合必须是有序序列**

示例

```C++
#include <vector>
#include <algorithm>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v1;
	vector<int> v2;
	for (int i = 0; i < 10; i++)
    {
		v1.push_back(i);
		v2.push_back(i+5);
	}

	vector<int> vTarget;
	//取两个里面较小的值给目标容器开辟空间
	vTarget.resize(min(v1.size(), v2.size()));

	//返回目标容器的最后一个元素的迭代器地址
	vector<int>::iterator itEnd = 
        set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), vTarget.begin());

	for_each(vTarget.begin(), itEnd, myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结

* 求交集的两个集合必须的有序序列
* 目标容器开辟空间需要从**两个容器中取小值**
* set_intersection返回值既是交集中最后一个元素的位置

#### 5.6.2 set_union

功能描述：求两个集合的并集

函数原型

- `set_union(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);  `
    - 求两个集合的并集
    - `beg1` 容器1开始迭代器
    - `end1` 容器1结束迭代器
    - `beg2` 容器2开始迭代器
    - `end2` 容器2结束迭代器
    - `dest` 目标容器开始迭代器

- **注意:两个集合必须是有序序列**

示例

```C++
#include <vector>
#include <algorithm>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v1;
	vector<int> v2;
	for (int i = 0; i < 10; i++) {
		v1.push_back(i);
		v2.push_back(i+5);
	}

	vector<int> vTarget;
	//取两个容器的和给目标容器开辟空间
	vTarget.resize(v1.size() + v2.size());

	//返回目标容器的最后一个元素的迭代器地址
	vector<int>::iterator itEnd = 
        set_union(v1.begin(), v1.end(), v2.begin(), v2.end(), vTarget.begin());

	for_each(vTarget.begin(), itEnd, myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结

- 求并集的两个集合必须的有序序列
- 目标容器开辟空间需要**两个容器相加**
- set_union返回值既是并集中最后一个元素的位置

#### 5.6.3  set_difference

功能描述：求两个集合的差集

函数原型

- `set_difference(iterator beg1, iterator end1, iterator beg2, iterator end2, iterator dest);  `
    - 求两个集合的差集
    - `beg1` 容器1开始迭代器
    - `end1` 容器1结束迭代器
    - `beg2` 容器2开始迭代器
    - `end2` 容器2结束迭代器
    - `dest` 目标容器开始迭代器

- **注意:两个集合必须是有序序列**

示例

```C++
#include <vector>
#include <algorithm>

class myPrint
{
public:
	void operator()(int val)
	{
		cout << val << " ";
	}
};

void test01()
{
	vector<int> v1;
	vector<int> v2;
	for (int i = 0; i < 10; i++) {
		v1.push_back(i);
		v2.push_back(i+5);
	}

	vector<int> vTarget;
	//取两个里面较大的值给目标容器开辟空间
	vTarget.resize( max(v1.size() , v2.size()));

	//返回目标容器的最后一个元素的迭代器地址
	cout << "v1与v2的差集为： " << endl;
	vector<int>::iterator itEnd = 
        set_difference(v1.begin(), v1.end(), v2.begin(), v2.end(), vTarget.begin());
	for_each(vTarget.begin(), itEnd, myPrint());
	cout << endl;


	cout << "v2与v1的差集为： " << endl;
	itEnd = set_difference(v2.begin(), v2.end(), v1.begin(), v1.end(), vTarget.begin());
	for_each(vTarget.begin(), itEnd, myPrint());
	cout << endl;
}

int main() {

	test01();

	system("pause");

	return 0;
}
```

总结

- 求差集的两个集合必须的有序序列
- 目标容器开辟空间需要从**两个容器取较大值**
- set_difference返回值既是差集中最后一个元素的位置