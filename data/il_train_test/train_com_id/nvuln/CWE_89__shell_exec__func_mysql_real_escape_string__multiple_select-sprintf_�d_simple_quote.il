var1 shell_exec
var1 mysql_real_escape_string var1
var2 sprintf SQL_QUERY var1
var3 mysql_connect
mysql_select_db
echo var2
var4 mysql_query var2
loop var5 mysql_fetch_array var4
  print_r var5
  echo
mysql_close var3
