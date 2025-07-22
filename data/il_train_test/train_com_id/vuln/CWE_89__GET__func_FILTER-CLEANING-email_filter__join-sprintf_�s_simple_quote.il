var1 GET
var2 filter_var var1 FILTER_SANITIZE_EMAIL
if filter_var var2 FILTER_VALIDATE_EMAIL
  var1 var2
else
  var1
var3 sprintf SQL_QUERY var1
var4 mysql_connect
mysql_select_db
echo var3
var5 mysql_query var3
loop var6 mysql_fetch_array var5
  print_r var6
  echo
mysql_close var4
