class class_name1
  public function function_name1
    return GET
var1 new class_name1
var2 var1->function_name1
var3 SQL_QUERY var2
var4 mysql_connect
mysql_select_db
echo var3
var5 mysql_query var3
loop var6 mysql_fetch_array var5
  print_r var6
  echo
mysql_close var4
