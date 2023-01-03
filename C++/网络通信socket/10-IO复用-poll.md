###### datetime:2022/12/30 17:11

###### author:nzb

# I/O复用-poll

## 4、poll

### 4.1、poll基本原理

-

poll和select在本质上没有差别，管理多个描述符也是送行轮询，根据描述符的状态进行处理，但是poll没有最大文件描述符数量的限制，它将用户传入的数组拷贝到内核空间，然后查询每个fd对应的设备状态，如果设备就绪则在设备等待队列中加入一项并继续遍历，如果遍历完所有fd后没有发现就绪设备，则挂起当前进程，直到设备就绪或者主动超时，被唤醒后它又要再次遍历fd。这个过程经历了多次无谓的遍历。

- select采用fdse煤用bitmap , poll采用了数组.

- poll和select同样存在一个缺点就是，文件描述符的数组植整体复制于 用户态和内核态的地址空间之间，而不论这些文件描述符是否有事件， 它的开销随着文件描述符数量的增加而线性增大.

- 还有poll返回后，也需要历遍整个描述符的数组才能得到有事件的描

### 4.2、poll基本流程

类似select

### 4.3、poll函数原型

```c++
#include <poll.h>
#include <arpa/inet.h>

int poll(struct pollfd *fds, unsigned int nfds, int timeout);

// （1）pollfd结构体定义如下：
struct pollfd {
    int fd;              /* 文件描述符 */
    short events;        /* 等待的事件 */
    short revents;       /* 实际发生了的事件 */
};
/*
  每一个pollfd结构体指定了一个被监视的文件描述符。因此可以传递多个结构体，指示poll()监视多个文件描述符。
 （2）events域是监视该文件描述符的事件掩码，由用户来设置这个域。
　　　　POLLIN　　　　　　　　 有数据可读。
　　　　POLLRDNORM　　　　　　有普通数据可读。
　　　　POLLRDBAND　　　　　　有优先数据可读。
　　　　POLLPRI　　　　　　　　有紧迫数据可读。
　　　　POLLOUT　　　　　　　　写数据不会导致阻塞。
　　　　POLLWRNORM　　　　　　写普通数据不会导致阻塞。
　　　　POLLWRBAND　　　　　　写优先数据不会导致阻塞。
　　　　POLLMSGSIGPOLL　　　　消息可用。
（3）revents域是文件描述符的操作结果事件掩码，内核在调用返回时设置这个域。events域中请求的任何事件都可能在revents域中返回。
　　 此外，revents域中还可能返回下列事件： 　　
　　　　POLLER　　  指定的文件描述符发生错误。
　　　　POLLHUP　　 指定的文件描述符挂起事件。
　　　　POLLNVAL　　指定的文件描述符非法。
　　 这些事件在events域中无意义，因为它们在合适的时候总是会从revents中返回。 　　
（4）举个栗子：要同时监视一个文件描述符是否可读和可写，
　　　　我们可以设置 events 为POLLIN | POLLOUT。
　　　　在poll返回时，我们可以检查revents中的标志，对应于文件描述符请求的events结构体。
　　　　如果POLLIN事件被设置，则文件描述符可以被读取而不阻塞。
　　　　如果POLLOUT被设置，则文件描述符可以写入而不导致阻塞。
　　　　这些标志并不是互斥的：它们可能被同时设置，表示这个文件描述符的读取和写入操作都会正常返回而不阻塞。
　　
（5）nfds参数是数组fds元素的个数。
（6）timeout参数指定等待的毫秒数，无论I/O是否准备好，poll都会返回。
　　　　timeout指定为负数值表示无限超时，使poll()一直挂起直到一个指定事件发生；
　　　　timeout为0指示poll调用立即返回并列出准备好I/O的文件描述符，但并不等待其它的事件。
　
（7）返回值和错误代码 　　
　　成功时，poll()返回结构体中revents域不为0的文件描述符个数；
　　如果在超时前没有任何事件发生，poll()返回0；
　　失败时，poll()返回-1，
　　　　并设置errno为下列值之一： 　　
　　　　EBADF　　      一个或多个结构体中指定的文件描述符无效。 　　
　　　　EFAULTfds　　  指针指向的地址超出进程的地址空间。 　　
　　　　EINTR　　　　   请求的事件之前产生一个信号，调用可以重新发起。 　　
　　　　EINVALnfds　　 参数超出PLIMIT_NOFILE值。 　　
　　　　ENOMEM　　     可用内存不足，无法完成请求。
*/
```

### 4.4、poll优点

没有最大连接数的限制。（基于链表来存储的）

### 4.5、poll缺点

- **时间复杂度**： 对socket进行扫描时是线性扫描，即采用轮询的方法，效率较低，时间复杂度O(n)。
  它将用户传入的数组拷贝到内核空间，然后查询每个fd对应的设备状态，如果设备就绪则在设备等待队列中加入一项并继续遍历，如果遍历完所有fd后没有发现就绪设备，则挂起当前进程，直到设备就绪或者主动超时，被唤醒后它又要再次遍历fd。这个过程经历了多次无谓的遍历。
- **内存拷贝**：大量的fd数组被整体复制于用户态和内核地址空间之间，而不管这样的复制是不是有意义。
- **水平触发**：如果报告了fd后，没有被处理，那么下次poll时会再次报告该fd。

> 注意：select和poll都需要在返回后，通过遍历文件描述符来获取已经就绪的socket。 事实上，同时连接的大量客户端在一时刻可能只有很少的处于就绪状态，因此随着监视的描述符数量的增长，其效率也会线性下降。

### 4.6、示例代码

tcpselect.cpp

```c++
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/fcntl.h>

// ulimit -n
#define MAXNFDS  1024

// 初始化服务端的监听端口。
int initserver(int port);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("usage: ./tcppoll port\n");
        return -1;
    }

    // 初始化服务端用于监听的socket。
    int listensock = initserver(atoi(argv[1]));
    printf("listensock=%d\n", listensock);

    if (listensock < 0) {
        printf("initserver() failed.\n");
        return -1;
    }

    int maxfd;   // fds数组中需要监视的socket的大小。
    struct pollfd fds[MAXNFDS];  // fds存放需要监视的socket。

    for (int ii = 0; ii < MAXNFDS; ii++) fds[ii].fd = -1; // 初始化数组，把全部的fd设置为-1。

    // 把listensock添加到数组中。
    fds[listensock].fd = listensock;
    fds[listensock].events = POLLIN;  // 有数据可读事件，包括新客户端的连接、客户端socket有数据可读和客户端socket断开三种情况。
    maxfd = listensock;

    while (1) {
        int infds = poll(fds, maxfd + 1, 5000);
        // printf("poll infds=%d\n",infds);

        // 返回失败。
        if (infds < 0) {
            printf("poll() failed.\n");
            perror("poll():");
            break;
        }

        // 超时。
        if (infds == 0) {
            printf("poll() timeout.\n");
            continue;
        }

        // 检查有事情发生的socket，包括监听和客户端连接的socket。
        // 这里是客户端的socket事件，每次都要遍历整个集合，因为可能有多个socket有事件。
        for (int eventfd = 0; eventfd <= maxfd; eventfd++) {
            if (fds[eventfd].fd < 0) continue;

            if ((fds[eventfd].revents & POLLIN) == 0) continue;

            fds[eventfd].revents = 0;  // 先把revents清空。

            if (eventfd == listensock) {
                // 如果发生事件的是listensock，表示有新的客户端连上来。
                struct sockaddr_in client;
                socklen_t len = sizeof(client);
                int clientsock = accept(listensock, (struct sockaddr *) &client, &len);
                if (clientsock < 0) {
                    printf("accept() failed.\n");
                    continue;
                }

                printf("client(socket=%d) connected ok.\n", clientsock);

                if (clientsock > MAXNFDS) {
                    printf("clientsock(%d)>MAXNFDS(%d)\n", clientsock, MAXNFDS);
                    close(clientsock);
                    continue;
                }

                fds[clientsock].fd = clientsock;
                fds[clientsock].events = POLLIN;
                fds[clientsock].revents = 0;
                if (maxfd < clientsock) maxfd = clientsock;

                printf("maxfd=%d\n", maxfd);
                continue;
            } else {
                // 客户端有数据过来或客户端的socket连接被断开。
                char buffer[1024];
                memset(buffer, 0, sizeof(buffer));

                // 读取客户端的数据。
                ssize_t isize = read(eventfd, buffer, sizeof(buffer));

                // 发生了错误或socket被对方关闭。
                if (isize <= 0) {
                    printf("client(eventfd=%d) disconnected.\n", eventfd);

                    close(eventfd);  // 关闭客户端的socket。

                    fds[eventfd].fd = -1;

                    // 重新计算maxfd的值，注意，只有当eventfd==maxfd时才需要计算。
                    if (eventfd == maxfd) {
                        for (int ii = maxfd; ii > 0; ii--) {
                            if (fds[ii].fd != -1) {
                                maxfd = ii;
                                break;
                            }
                        }

                        printf("maxfd=%d\n", maxfd);
                    }

                    continue;
                }

                printf("recv(eventfd=%d,size=%d):%s\n", eventfd, isize, buffer);

                // 把收到的报文发回给客户端。
                write(eventfd, buffer, strlen(buffer));
            }
        }
    }

    return 0;
}

// 初始化服务端的监听端口。
int initserver(int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        printf("socket() failed.\n");
        return -1;
    }

    // Linux如下
    int opt = 1;
    unsigned int len = sizeof(opt);
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, len);
    setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &opt, len);

    struct sockaddr_in servaddr;
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0) {
        printf("bind() failed.\n");
        close(sock);
        return -1;
    }

    if (listen(sock, 5) != 0) {
        printf("listen() failed.\n");
        close(sock);
        return -1;
    }

    return sock;
}
```

client.cpp

```c++
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("usage:./tcpclient ip port\n");
        return -1;
    }

    int sockfd;
    struct sockaddr_in servaddr;
    char buf[1024];

    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("socket() failed.\n");
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(atoi(argv[2]));
    servaddr.sin_addr.s_addr = inet_addr(argv[1]);

    if (connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) != 0) {
        printf("connect(%s:%s) failed.\n", argv[1], argv[2]);
        close(sockfd);
        return -1;
    }

    printf("connect ok.\n");

    for (int ii = 0; ii < 10000; ii++) {
        // 从命令行输入内容。
        memset(buf, 0, sizeof(buf));
        printf("please input:");
        scanf("%s", buf);
        // sprintf(buf,"1111111111111111111111ii=%08d",ii);

        if (write(sockfd, buf, strlen(buf)) <= 0) {
            printf("write() failed.\n");
            close(sockfd);
            return -1;
        }

        memset(buf, 0, sizeof(buf));
        if (read(sockfd, buf, sizeof(buf)) <= 0) {
            printf("read() failed.\n");
            close(sockfd);
            return -1;
        }

        printf("recv:%s\n", buf);

        // close(sockfd); break;
    }
} 
```