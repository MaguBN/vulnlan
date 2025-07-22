var1
exec var1 var2 var3
var4 var2_0
var5 filter_var var4 FILTER_SANITIZE_EMAIL
if filter_var var5 FILTER_VALIDATE_EMAIL
  var4 var5
else
  var4
var6 SQL_QUERY var4
var7 mysql_connect
mysql_select_db
echo var6
var8 mysql_query var6
loop var9 mysql_fetch_array var8
  print_r var9
  echo
mysql_close var7
