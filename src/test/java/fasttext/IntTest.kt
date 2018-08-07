package fasttext

import com.sun.tools.hat.internal.model.JavaLong
import java.math.BigInteger
import java.util.*

//read word^ hash 3675003649
// int32 -619963647
// uint64 18446744073089587969 wid 1
fun main(args: Array<String>) {

    val i = 3675003649L
    val int :Int = i.toInt()
    //val b = BigInteger(i.toString())

    println(i.toString(2))
    println(i)
    println(int)
    println(BigInteger("100100111100111110010011111111",2))
    println(BigInteger("18446744073089587969").toString(2))
    println(BigInteger(i.toString(2).padStart(64,'1'),2))
    var b = BigInteger(i.toString(2).padStart(64,'1'),2)
    println(b)
    println(b.bitLength())
    val x = BigInteger("1".repeat(64),2)



}