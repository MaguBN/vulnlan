var1 GET
var2 sprintf SQL_QUERY var1
var3 mysql_connect
mysql_select_db
echo var2
var4 mysql_query var2
loop var5 mysql_fetch_array var4
  print_r var5
  echo
mysql_close var3
