var1 popen
var2 fread var1 4096
pclose var1
var3 filter_var var2 FILTER_SANITIZE_FULL_SPECIAL_CHARS
var2 var3
var4 SQL_QUERY var2
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
