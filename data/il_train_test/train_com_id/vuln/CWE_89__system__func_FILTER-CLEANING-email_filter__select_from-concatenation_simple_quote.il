var1 system var2
var3 filter_var var1 FILTER_SANITIZE_EMAIL
if filter_var var3 FILTER_VALIDATE_EMAIL
  var1 var3
else
  var1
var4 SQL_QUERY var1
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
