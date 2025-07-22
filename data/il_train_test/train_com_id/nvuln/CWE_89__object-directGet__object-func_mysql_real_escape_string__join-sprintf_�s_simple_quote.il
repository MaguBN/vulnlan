class class_name1
  public function function_name1
    return GET
var1 new class_name1
var2 var1->function_name1
class class_name2
  public function function_name2 var3
    return mysql_real_escape_string var3
var1 new class_name2
var2 var1->function_name2 var2
var4 sprintf SQL_QUERY var2
var5 mysql_connect
mysql_select_db
echo var4
var6 mysql_query var4
loop var7 mysql_fetch_array var6
  print_r var7
  echo
mysql_close var5
