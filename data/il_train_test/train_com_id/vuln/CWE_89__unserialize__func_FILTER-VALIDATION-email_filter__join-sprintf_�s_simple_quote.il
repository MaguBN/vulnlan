var1 POST
var2 unserialize var1
if filter_var var3 FILTER_VALIDATE_EMAIL
  var2 var3
else
  var2
var4 sprintf SQL_QUERY var2
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
