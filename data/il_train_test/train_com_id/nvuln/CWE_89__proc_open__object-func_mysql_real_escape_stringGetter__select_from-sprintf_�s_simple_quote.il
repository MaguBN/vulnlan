var1 array
0 array
1 array
2 array
var2
var3 proc_open var1 var4 var2 NULL
if is_resource var3
  fclose var4_0
  var5 stream_get_contents var4_1
  fclose var4_1
  var6 proc_close var3
class class_name1
  private var7
  public function function_name1 var8
    var9->data var8
  public function function_name2
    return var9->data
  public function function_name3
    var9->data mysql_real_escape_string var9->data
var10 new class_name1 var5
var10->function_name3
var5 var10->function_name2
var11 sprintf SQL_QUERY var5
var12 mysql_connect
mysql_select_db
echo var11
var13 mysql_query var11
loop var7 mysql_fetch_array var13
  print_r var7
  echo
mysql_close var12
