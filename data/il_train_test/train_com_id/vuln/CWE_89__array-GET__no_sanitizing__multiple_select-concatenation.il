var1 array
var1_0
var1_1 GET
var1_2
var2 var1_1
var3 SQL_QUERY var2
var4 mysql_connect
mysql_select_db
echo var3
var5 mysql_query var3
loop var6 mysql_fetch_array var5
  print_r var6
  echo
mysql_close var4
