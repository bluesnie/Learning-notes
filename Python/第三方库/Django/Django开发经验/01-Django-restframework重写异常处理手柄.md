###### datetime:2020/1/8 16:02
###### author:nzb

## 重写异常处理返回
例如：
- 正常返回
```json
    {
        "detail": "方法 “GET” 不被允许。"
    }
```
- 重写返回
```json
    {
        "code": 10001,
        "errMsg": "方法 “GET” 不被允许。"
    }
```

### 源码流程

- `dispath()` 分发
```python
    def dispatch(self, request, *args, **kwargs):
        """
        `.dispatch()` is pretty much the same as Django's regular dispatch,
        but with extra hooks for startup, finalize, and exception handling.
        """
        self.args = args
        self.kwargs = kwargs
        request = self.initialize_request(request, *args, **kwargs)
        self.request = request
        self.headers = self.default_response_headers  # deprecate?

        try:
            self.initial(request, *args, **kwargs)

            # Get the appropriate handler method
            if request.method.lower() in self.http_method_names:
                handler = getattr(self, request.method.lower(),
                                  self.http_method_not_allowed)
            else:
                handler = self.http_method_not_allowed
            # 活动到某个视图的某个方法，例如create()
            # 视图函数里的serializer.is_valid(raise_exception=True), 
            # 会去验证序列化里面自带的验证规则和自定义的验证规则（并抛出异常）							
            response = handler(request, *args, **kwargs)    

        except Exception as exc:    
            # 捕获异常
            response = self.handle_exception(exc)

        self.response = self.finalize_response(request, response, *args, **kwargs)
        return self.response
```


- `handle_exception()`

```python
    # 捕获异常后执行
    def handle_exception(self, exc):
        """
        Handle any exception that occurs, by returning an appropriate response,
        or re-raising the error.
        """
        if isinstance(exc, (exceptions.NotAuthenticated,
                            exceptions.AuthenticationFailed)):
            # WWW-Authenticate header for 401 responses, else coerce to 403
            auth_header = self.get_authenticate_header(self.request)

            if auth_header:
                exc.auth_header = auth_header
            else:
                exc.status_code = status.HTTP_403_FORBIDDEN
        # 获取重写的异常返回函数
        exception_handler = self.get_exception_handler()

        context = self.get_exception_handler_context()
        response = exception_handler(exc, context)

        if response is None:
            self.raise_uncaught_exception(exc)

        response.exception = True
        return response
    
    # 获取配置文件
    def get_exception_handler(self):
        """
        Returns the exception handler that this view uses.
        """
        return self.settings.EXCEPTION_HANDLER
```

- `重写函数custom_exception_handler()`

```python

from rest_framework.views import exception_handler
def custom_exception_handler(exc,context):
    """
    框架自带错误码(常见已知的)
                (   (400, "invalid"),
                    (401, "authentication_failed"),
                    (401, "not_authenticated"),
                    (403, "permission_denied"),
                    (404, "not_found"),
                    (405, "method_not_allowed"),
               )
     不常见的
               (    (400, "parse_error"),
                    (406, "not_acceptable"),
                    (415, "unsupported_media_type"),
                    (429, "throttled")
               )
    """

    response = exception_handler(exc, context)  # 获取本来应该返回的exception的response
    request = context.get("request", None)
    ...


    if response is not None:
        if response.status_code == 403:     # 权限
            pass
        elif response.status_code == 401:   # 是否登录
            pass
        elif response.status_code == 404:   # 资源未找到
            response.data['code'] = org_status_code.NOTFOUNDERROR_CODE.get("code", None)
            response.data['errMsg'] = org_status_code.NOTFOUNDERROR_CODE.get("detail", None)
            del response.data['detail']
        elif response.status_code == 405:  # 方法不允许
            pass
        elif response.status_code == 400:   # 重写django自带的序列化错误以及自定义的序列化错误
            response = process_400_BAD_REQUEST(response)
        else:
            pass

    return response


def process_400_BAD_REQUEST(response):
    """映射自带错误和返回自定义错误"""
    old_data = response.data
    for k, v in old_data.items():
        new_data = {}
        # 框架自带错误码(常见已知的)
        if v[0].code == "required":
            new_data['code'] = org_status_code.NOTNULL_CODE.get("code", None)
            new_data['errMsg'] = org_status_code.NOTNULL_CODE.get("detail", None).format(k)

        elif v[0].code == "invalid":
            new_data['code'] = org_status_code.TYPEERROR_CODE.get("code", None)
            new_data['errMsg'] = org_status_code.TYPEERROR_CODE.get("detail", None).format(k, v[0])

        elif v[0].code == "incorrect_type":  # 类型错误
            new_data['code'] = org_status_code.TYPEERROR_CODE.get("code", None)
            new_data['errMsg'] = org_status_code.TYPEERROR_CODE.get("detail", None).format(k, v[
                0])  # "{0}字段{1}".format(k, v[0])

        elif v[0].code == "does_not_exist":  # 外键对象不存在
            new_data['code'] = org_status_code.FOREIGNKEYNOEXISTED_CODE.get("code", None)
            new_data['errMsg'] = org_status_code.FOREIGNKEYNOEXISTED_CODE.get("detail", None).format(k, v[
                0])  # "{0}字段{1}".format(k, v[0])

        elif v[0].code == "unique":  # 已存在
            new_data['code'] = org_status_code.EXISTED_CODE.get("code", None)
            new_data['errMsg'] = org_status_code.EXISTED_CODE.get("detail", None)

        elif v[0].code == "max_length":  # 最大长度
            new_data['code'] = org_status_code.MAXLENGTHERROR_CODE.get("code", None)

            new_data['errMsg'] = v[0].replace("这个", k)

        elif v[0].code == "min_length":  # 最小长度
            new_data['code'] = org_status_code.MINLENGTHERROR_CODE.get("code", None)
            new_data['errMsg'] = v[0].replace("这个", k)

        # 自定义错误码以及未知的错误码
        else:
            if isinstance(v[0].code, int):  # 自定义的错误
                new_data['code'] = v[0].code
                new_data['errMsg'] = v[0]
            else:
                response.data = old_data
                return response
        response.data = new_data
        return response
```

- `settings.py配置`

```python
REST_FRAMEWORK = {
    'DATETIME_FORMAT': '%Y/%m/%d %H:%M:%S',
    'JWT_ALLOW_REFRESH': True,
    'DEFAULT_AUTHENTICATION_CLASSES': (
        "rest_framework_jwt.authentication.JSONWebTokenAuthentication",
        # 'utils.authentication.CustomAuthenticate',  # 自定义 JSON Token Authentication
        # 'rest_framework.authentication.BasicAuthentication',
        # 'rest_framework.authentication.SessionAuthentication',
    ),
    'EXCEPTION_HANDLER': 'utils.exceptions.custom_exception_handler',   # 自定义重写的异常处理返回
    'SEARCH_PARAM': 'kw',
}
```