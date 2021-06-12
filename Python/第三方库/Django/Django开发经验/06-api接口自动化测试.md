###### datetime:2020/9/17 14:35
###### author:nzb

# [api接口自动化测试](https://q1mi.github.io/Django-REST-framework-documentation/api-guide/testing_zh/)

- 配置文件数据库配置

```python
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': 'sign_server',
            'USER': 'root',
            'PASSWORD':'123456',
            'PORT': 3306,
            'HOST': '127.0.0.1',
            'TEST':{                # 测试数据库，每次测试都会自动创建，测试完后会自动删除
                    'NAME': 'test_sign_server',
                    'CHARSET': 'utf8mb4',
                    'COLLATION': 'utf8mb4_general_ci'
            },
            'OPTIONS': {
                'charset': 'utf8mb4'
            }
        }
    }
```

- 测试示例

```python
    from django.test import TestCase
    import json
    from pprint import pprint
    from django.urls import reverse
    from rest_framework import status
    from rest_framework.test import APITestCase
    from sign.models import SignInfo
    
    class SignTests(APITestCase):
    
        def test_sign(self):
            """签到"""
            print("开始测试")
            url = reverse('sign-list')
            # url：/api/sign/
            for user_id in range(1, 21):
                data = {'pro_id': 1, "obj_id": 1, "obj_type": 0, "user_id": user_id, "sign_type": 0}
                response = self.client.post(url, data, format='json')
                self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
                self.assertEqual(response.data.get("code"), 10000, response.data)
    
            # 用户1第二条记录
            data = {'pro_id': 1, "obj_id": 2, "obj_type": 0, "user_id": 1, "sign_type": 0}
            response = self.client.post(url, data, format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
            self.assertEqual(response.data.get("code"), 10000, response.data)
    
            ins_id = response.data.get("results", {}).get("id")
    
            # 详情
            url = reverse('sign-detail', args=[ins_id])
            # url：/api/sign/21/
            response = self.client.get(url)
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
            self.assertEqual(response.data.get("code"), 10000, response.data)
            self.assertEqual(response.data.get("results", {}).get("status", None), 0, "用户签到状态错误")
    
            # 修改处理状态
            url = reverse('sign-detail', args=[ins_id])
            # url：/api/sign/21/
            data = {"status": 2}
            response = self.client.put(url, data, format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
            self.assertEqual(response.data.get("code"), 10000, response.data)
            self.assertEqual(SignInfo.objects.get(id=ins_id).status, 2, "修改处理状态出错")
    
            # 补签(用户2在补签一个， obj_id=2)
            url = reverse('sign-list')
            # url：/api/sign/
            data = {'pro_id': 1, "obj_id": 2, "obj_type": 0, "user_id": 2, "sign_type": 1,
                    "extra4": "我要补签", "extra_explain": json.dumps({"extra1": "扩展1说明", "extra2": None, "extra3": None,
                                                                   "extra4": "补签说明", "extra5": None, "extra6": None})}
            response = self.client.post(url, data, format='json')
    
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
            self.assertEqual(response.data.get("code"), 10000, response.data)
            self.assertEqual(response.data.get("results", {}).get("sign_type", None), 1, "补签失败")
            self.assertEqual(response.data.get("results", {}).get("extra4", None), "我要补签", "补签的额外字段错误")
            self.assertEqual(SignInfo.objects.filter().count(), 22, "总数量不对")
    
            # 用户1的签到历史
            url = reverse('sign-list') + "?offset=0&limit=1&user_id=1"
            # url：/api/sign/?offset=0&limit=1&user_id=1
            response = self.client.get(url)
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
            self.assertEqual(response.data.get("code"), 10000, response.data)
            self.assertEqual(response.data.get("count"), 2, "用户1的数量不对")
    
            # 签到扩展字段说明
            url = reverse('sign-detail', args=['extra_explain']) + "?pro_id=1&obj_id=2&obj_type=0"
            # url：/api/sign/extra_explain/?pro_id=1&obj_id=2&obj_type=0
            response = self.client.get(url)
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.data)
            self.assertEqual(response.data.get("code"), 10000, response.data)
            self.assertEqual(len(response.data.get("results", [])), 1, "额外参数说明错误")
```

- [路由的反向解析](../06-Django的路由系统.md#命名URL和URL反向解析)

其中用drf注册的路由系统，反向解析如下

- 源码

```python
routes = [
        # List route.
        Route(
            url=r'^{prefix}{trailing_slash}$',
            mapping={
                'get': 'list',       # 列表
                'post': 'create'     # 创建
            },
            name='{basename}-list',  # 本例中：sign-list， basename：url中的别名
            detail=False,
            initkwargs={'suffix': 'List'}
        ),
        # Dynamically generated list routes. Generated using
        # @action(detail=False) decorator on methods of the viewset.
        DynamicRoute(
            url=r'^{prefix}/{url_path}{trailing_slash}$',
            name='{basename}-{url_name}',               # 装饰器的接口：detail=False
            detail=False,
            initkwargs={}
        ),
        # Detail route.
        Route(
            url=r'^{prefix}/{lookup}{trailing_slash}$',
            mapping={
                'get': 'retrieve',          # 详情
                'put': 'update',            # 更新
                'patch': 'partial_update',  # 更新
                'delete': 'destroy'         # 删除
            },
            name='{basename}-detail',       # 本例中：sign-detail， basename：url中的别名
            detail=True,
            initkwargs={'suffix': 'Instance'}
        ),
        # Dynamically generated detail routes. Generated using
        # @action(detail=True) decorator on methods of the viewset.
        DynamicRoute(
            url=r'^{prefix}/{lookup}/{url_path}{trailing_slash}$',
            name='{basename}-{url_name}',               # 装饰器的接口：detail=False
            detail=True,
            initkwargs={}
        ),
    ]
```
- 注意
    - 被action修饰的接口尽量不用下划线连接，否则reverse不能反向解析，如果使用了下划线，只能这样`url = reverse('sign-detail', args=['extra_explain']) + "?pro_id=1&obj_id=2&obj_type=0"`
    - detail=True：`reverse("sign-extra", args=[1])`
        - 结果：`/api/sign/extra/`
    - detail=False：`reverse("sign-extra")`
        - 结果：`/api/sign/1/extra/`
    - 如果测试的接口有用到其他表信息：比如用户表可以直接用表创建用户
