var1 GET
var2 filter_var var1 FILTER_SANITIZE_FULL_SPECIAL_CHARS
var1 var2
var3 SQL_QUERY var1
var4 mysql_connect
mysql_select_db
echo var3
var5 mysql_query var3
loop var6 mysql_fetch_array var5
  print_r var6
  echo
mysql_close var4
