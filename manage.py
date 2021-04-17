#!/usr/bin/python
# -*- coding:utf-8 -*-
# !/usr/bin/env python
import urllib

import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.web import RequestHandler
import py_eureka_client.eureka_client as eureka_client
from tornado.options import define, options
import py_eureka_client.netint_utils as netint_utils
from time import sleep

define("port", default=3333, help="run on the given port", type=int)


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        username = self.get_argument('username', 'Hello')
        self.write(username + ', Administrator User!')

    def post(self):
        username = self.get_argument('username', 'Hello')
        self.write(username + ', Administrator User!')


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        username = self.get_argument('username', 'Hello')
        self.write(username + ', Coisini User!')

    def post(self):
        username = self.get_argument('username', 'Hello')
        self.write(username + ', Coisini User!')




def main():
    tornado.options.parse_command_line()
    # 注册eureka服务
    eureka_client.init(eureka_server="http://localhost:9000/eureka/",
                                       app_name="python-service",
                                       instance_port=3333)
    app = tornado.web.Application(handlers=[(r"/test", IndexHandler), (r"/main", MainHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

    try:
        res = eureka_client.do_service("RIBBON-PROVIDER", "/hello?name=python", return_type="string")
        print("result of other service" + res)
    except urllib.request.HTTPError as e:
        # If all nodes are down, a `HTTPError` will raise.
        print(e)
    # eureka_client.do_service_async("ribbon-provider", "/hello?name=python", on_success=success_callabck,
    #                                on_error=error_callback)

if __name__ == '__main__':
    main()