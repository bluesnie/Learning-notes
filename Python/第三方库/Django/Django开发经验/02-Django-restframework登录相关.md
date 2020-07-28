###### datetime:2020/5/13 10:33
###### author:nzb

## 登录相关（基于jwt token登录；单个用户表或多个用户表）

- urls.py
```python
    path("login/", CustomLoginJSONWebToken.as_view()),
```

- settings.py

```python
# 验证中间件
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    # 'corsheaders.middleware.CorsPostCsrfMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'middlewares.ValidTokenMiddleware',     # 换台设备需重新登录
]

# drf框架的配置信息
REST_FRAMEWORK = {
    # 默认分页
    'DEFAULT_PAGINATION_CLASS': 'utils.PagePagination',
    'DATETIME_FORMAT': '%Y-%m-%d %H:%M:%S',
    # 自定义token内含值
    'JWT_PAYLOAD_HANDLER': 'utils.custom_payload_handler',
    # 异常处理
    'EXCEPTION_HANDLER': 'utils.exception_handler.exception_handler',       # 见01-重写异常处理手柄
    # 用户登陆认证方式
    'DEFAULT_AUTHENTICATION_CLASSES': (
    #单个用户表配置
        # 'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
        # 'rest_framework.authentication.SessionAuthentication',
        # 'rest_framework.authentication.BasicAuthentication',
    # 多个用户表配置
    'authentication.CustomAuthenticate',  # 自定义 JSON Token Authentication
        
    ),
    # 获取用户的secret_key
    'JWT_GET_USER_SECRET_KEY': 'utils.jwt_get_user_secret',

}
# jwt载荷中的有效期设置(from rest_framework_jwt)
token_time =  datetime.timedelta(days=365)
JWT_AUTH = {
    'JWT_EXPIRATION_DELTA': token_time,             # 有效期设置
    'JWT_REFRESH_EXPIRATION_DELTA': token_time,     # 刷新有效期
    'JWT_RESPONSE_PAYLOAD_HANDLER': '.utils.custom_jwt_response_payload_handler',   # 自定义返回认证通过后的数据
}
```

- utils.py

```python
from django.contrib.auth import authenticate, get_user_model
from rest_framework_jwt.utils import jwt_encode_handler
from rest_framework_jwt.settings import api_settings


def custom_payload_handler(user):
    '''自定义token内含值
    :param user:  user auth model
    :return: 计算token的基本信息
    '''

    # jwt token payload的基本信息： user_id 用户主键id, 用户first_name, 用户电话号码 phone
    payload = {
        'user_id': user.pk,
        'username': user.username,
        'exp': datetime.datetime.utcnow() + api_settings.JWT_EXPIRATION_DELTA,
        # 'phone': user.phone
        'user_secret': str(uuid.uuid4())    #　uuid
    }

    if api_settings.JWT_ALLOW_REFRESH:
        payload['orig_iat'] = timegm(
            datetime.datetime.utcnow().utctimetuple()
        )

    if api_settings.JWT_AUDIENCE is not None:
        payload['aud'] = api_settings.JWT_AUDIENCE

    if api_settings.JWT_ISSUER is not None:
        payload['iss'] = api_settings.JWT_ISSUER

    return payload

def generate_user_token(user):
    """生成用户token"""
    user_model = get_user_model()

    payload = custom_payload_handler(user)
    token = jwt_encode_handler(payload)

    return token

def custom_jwt_response_payload_handler(token, user=None):
    """
    自定义jwt认证成功返回的数据
    :token  返回的jwt
    :user   当前登录的用户信息[对象]
    :request 当前本次客户端提交过来的数据

    Example:
        return {
            'token': token,
            'user': UserSerializer(user, context={'request': request}).data
        }
    """
    data = {
        'code': 10000,
        'results': {
            'token': token,
            'id': user.id,
            'username': user.username,
            ．．．．
        }
    }
    return data
    
def jwt_get_secret_key(payload=None):
    """获取用户的secret_key（例如uuid）
    For enhanced security you may want to use a secret key based on user.

    This way you have an option to logout only this user if:
        - token is compromised
        - password is changed
        - etc.
    """
    return user.user_secret
    
# 分页
class PagePagination(LimitOffsetPagination):
    """分页"""
    # page_size = 1
    limit_query_param = 'limit'
    offset_query_param = 'offset'
    max_limit = 20

    def get_paginated_response(self, data):
        ret = dict([
            ('code', 10000,
            ('errMsg', ''),
            ('count', self.count),
            ('previous', self.get_previous_link()),
            ('next', self.get_next_link()),
        ])
        if isinstance(data, dict):
            ret.update(**data)
        else:
            ret.update({
                'results': data
            })
        return Response(OrderedDict(ret))
```

- views.py

```python
from rest_framework_jwt.views import JSONWebTokenAPIView
from rest_framework_jwt.settings import api_settings

from utils import custom_jwt_response_payload_handler

class CustomLoginJSONWebToken(JSONWebTokenAPIView):
    """
    自定义登录
    """
    serializer_class = CustomJSONWebTokenSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)

        if serializer.is_valid():
            user = serializer.object.get('user') or request.user
            token = serializer.object.get('token')
            #　可自定义返回认证成功后的数据,settings中的JWT_AUTH中的JWT_RESPONSE_PAYLOAD_HANDLER设置
            # 这里还可以写需要的相应逻辑
            response_data = custom_jwt_response_payload_handler(token, user, request)
            response = Response(response_data)
            if api_settings.JWT_AUTH_COOKIE:
                expiration = (datetime.utcnow() +
                              api_settings.JWT_EXPIRATION_DELTA)
                response.set_cookie(api_settings.JWT_AUTH_COOKIE,
                                    token,
                                    expires=expiration,
                                    httponly=True)
            return response

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
```

- serializers.py

```python
from rest_framework_jwt.serializers import JSONWebTokenSerializer
from django.contrib.auth import authenticate, get_user_model
from utils import generate_user_token

class CustomJSONWebTokenSerializer(JSONWebTokenSerializer):
    """自定义账号密码登录序列号"""

    def validate(self, attrs):
        # 密码账户验证
        username = attrs.get(self.username_field)
        password = attrs.get('password', None)

        credentials = {
            self.username_field: username,
            'password': password
        }
        if all(credentials.values()):
            user = authenticate(**credentials)

            if user:
                if not user.is_active:
                    msg = {'code': 10996, 'errMsg': '用户被冻结'}
                    raise serializers.ValidationError(msg)

                # payload = jwt_payload_handler(user)

                return {
                    # 'token': jwt_encode_handler(payload),
                    'token': generate_user_token(user),
                    'user': user
                }
            else:
                # 若认证失败
                msg = {'code': 10991, 'errMsg': '用户名和密码错误'}
                raise serializers.ValidationError(msg)
        else:
            msg = {'code': 10997, 'errMsg': 'username和password为必填字段'}
            raise serializers.ValidationError(msg)


from rest_framework_jwt.serializers import VerifyJSONWebTokenSerializer

class CustomVerifyJSONWebTokenSerializer(VerifyJSONWebTokenSerializer):
    """
    多个用户表时自定义验证
    中间件中使用
    """
    def _check_user(self, payload):
        # print("-------自定义验证的payload：", payload)
        username = jwt_get_username_from_payload(payload)

        if not username:
            msg = _('Invalid payload.')
            raise serializers.ValidationError(msg)

        # Make sure user exists
        token_type = payload.get("type", None)
        if token_type == "manager":
            try:
            # 表1
                user = UserManager.objects.get_by_natural_key(username)
            except UserManager.DoesNotExist:
                msg = _("User doesn't exist.")
                raise serializers.ValidationError(msg)
            if not user.is_active:
                msg = _('User account is disabled.')
                raise serializers.ValidationError(msg)
        else:
            try:
            # 表2
                user = UserStuInfo.objects.filter(id=payload.get("user_id", None), sno=payload.get("sno", None), is_del=False).first()
            except UserStuInfo.DoesNotExist:
                msg = _("User doesn't exist.")
                raise serializers.ValidationError(msg)

        return user

```

- authentication.py（多个用户表时的登录验证）

```python
from rest_framework_jwt.authentication import JSONWebTokenAuthentication, jwt_get_username_from_payload


class CustomAuthenticate(JSONWebTokenAuthentication):
    """多个用户表自定义设置登录选项"""

    def authenticate_credentials(self, payload):
        """
        Returns an active user that matches the payload's user id and email.
        """
        # User = get_user_model()
        username = jwt_get_username_from_payload(payload)

        # if

        if not username:
            msg = _('Invalid payload.')
            raise exceptions.AuthenticationFailed(msg)
        token_type = payload.get("type", None)
        if token_type == "manager":
            try:
                # 举例用户表1
                user = UserManager.objects.get_by_natural_key(username)
            except user.DoesNotExist:
                msg = _('Invalid signature.')
                raise exceptions.AuthenticationFailed(msg)
            if not user.is_active:
                msg = _('User account is disabled.')
                raise exceptions.AuthenticationFailed(msg)
        else:
            try:
                # 举例用户表2
                user = UserStuInfo.objects.filter(id=payload.get("user_id", None), sno=payload.get("sno", None), is_del=False).first()
            except user.DoesNotExist:
                msg = _('Invalid signature.')
                raise exceptions.AuthenticationFailed(msg)
        return user
```

- middleware.py

```python
from uuid import uuid4
import json

from django.http import HttpResponse
from jwt import InvalidSignatureError
from rest_framework.exceptions import ValidationError
from django.utils.deprecation import MiddlewareMixin
from rest_framework_jwt.utils import jwt_decode_handler

from authentication import CustomVerifyJSONWebTokenSerializer

# 1.每次登录 response 处理 记录 jwt
# 2.每次请求判断 jwt是否与表中相等(相当于用户异设备登录获取了新的jwt)  不等 就修改uuid


class ValidTokenMiddleware(MiddlewareMixin):

    def process_request(self, request):
        # 用于处理 所有带 jwt 的请求
        jwt_token = request.META.get('HTTP_AUTHORIZATION', None)
        if jwt_token is not None and jwt_token != '':
            data = {
                'token': request.META['HTTP_AUTHORIZATION'].split(' ')[1],
            }
            try:
                # valid_data = VerifyJSONWebTokenSerializer().validate(data)        # 原来的
                valid_data = CustomVerifyJSONWebTokenSerializer().validate(data)    # 多用户自定义后的
                user = valid_data['user']
                # if user:
                    # print("------------请求时用户的uuid:{0}".format(user.user_secret))
            except (InvalidSignatureError, ValidationError):
                # 找不到用户
                data = json.dumps({"code": 10000, "errMsg": "用户未登录"})
                return HttpResponse(data, content_type='application/json', status=400)
            # if user.user_jwt != data['token']:
            decode_token = jwt_decode_handler(data['token'])        # 解析token，这里面就有获取用户的user_secret，所以需要重写jwt_get_secret_key
            # print("------------请求时带的token：{0}".format(decode_token))
            if not user:
                data = json.dumps({"code": 10000, "errMsg": "用户未登录"})
                return HttpResponse(data, content_type='application/json', status=400)
            elif str(user.user_secret) != decode_token.get("user_secret"):  
                user.user_secret = uuid4()
                user.save()
                data = json.dumps({"code": 10000, "errMsg": "用户未登录"})
                return HttpResponse(data, content_type='application/json', status=400)

    def process_response(self, request, response):
        # 仅用于处理 login请求
        # print("----------", request.META['PATH_INFO'])
        
        MANAGER_LOGIN_PATH = "/api/teacher/login/"
        STUDENT_LOGIN_PATH = "/api/student/login/"
        
        if request.META['PATH_INFO'] in (MANAGER_LOGIN_PATH, STUDENT_LOGIN_PATH):
            try:
                rep_data = response.data
            except AttributeError as e:
                print("报错信息：", e.args)
            results = rep_data.get('results', None)
            if results:
                # valid_data = VerifyJSONWebTokenSerializer().validate(results)         # 原来的
                valid_data = CustomVerifyJSONWebTokenSerializer().validate(results)     # 多用户自定义后的
                user = valid_data['user']
                # user.user_jwt = rep_data['results']['token']
                decode_token = jwt_decode_handler(rep_data['results']['token'])
                # print("--------登录后的token:", decode_token)
                user.user_secret = decode_token.get("user_secret")
                user.save()
                return response
            else:
                return response
        else:
            return response
```
