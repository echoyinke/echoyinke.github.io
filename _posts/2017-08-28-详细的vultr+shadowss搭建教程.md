

****

###  1、注册Paypal

以下以带visa标识的信用卡为例，其他卡等同, 选Paypal是可以避免信用卡盗刷，万一被盗刷，Paypal的赔付速度也是很快的

注册[Paypal](https://paypal.com)时选择买家(外币转换手续费不用你支付，由卖家支付)，绑定信用卡/借记卡

在我的账户中上方可以看到账户状态，显示未认证(找不到的用ctrl+f搜索“未认证”)，进行认证

绑定你的信用卡/借记卡，Paypal会从你的信用卡/借记卡扣除小额的费用，等2-3天美元账单入账后，你会从入账单中看到4 位数的 code，在Paypal中输入4 位数的 code就认证完成了

注意：若绑定的是银联借记卡，则需要：

```
点击右上角的齿轮(设置)->点击 上方的付款->点击 管理预核准付款->点击 设置可用资金来源

->勾选该银联借记卡->点击 兑换选项->选择 在给我的账单中使用卖家列出的币种->提交
```

### 注册Vultr(充多少送多少活动)

点击  [优惠链接](https://www.vultr.com/?ref=7206675)  (充值$10，返现$10)

进入官网，点Creat Account注册。

### 2、部署：

支付完毕后，回到Vulrt的界面，点击右侧的＋号，部署(deploy)一个新的VPS(虚拟服务器)。建议选择美国LA节点，操作系统选择 federa，套餐选$2.5/月或者$5/月的。点生成，之后跳转到管理页面，当显示绿色的running时，该VPS就部署好了，点击该VPS，进入管理页面，可以看到VPS的**IP**，在VPS管理页面点那个眼睛，可以看到你的ROOT账户的**密码**。

### 3、服务器端操作：

```
ssh root@IP_address
#利用SSH登陆 ROOT账户，然后输入yes，输入密码，密码在此台vps管理页面的Password:
#点击后面两个正方形，复制密码，粘贴到terminal_1(密码不会显示)，直接回车。

iptables -F
#移除firewall, 一般vps都有防火墙

rm -rf /etc/localtime; ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
#把vps时间设定成北京时间，这点很重要，出了问题，查日志方便
#用命令 date 查看
```

​	剩下的步骤：https://github.com/echoyinke/shadowsocks

### 4、客户端操作：

​	根据自己的机型下载不同的客户端：https://github.com/echoyinke/shadowsocks

