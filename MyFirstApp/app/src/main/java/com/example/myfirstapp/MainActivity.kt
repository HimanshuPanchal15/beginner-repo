package com.example.myfirstapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        val btnClickMe = findViewById<Button>(R.id.mybutton)
        btnClickMe.text = "Click Me!"
//        btnClickMe.setOnClickListener {btnClickMe.text="Haha you clicked me!"}

        val textClickMe = findViewById<TextView>(R.id.textView)
//        textClickMe.setOnClickListener { textClickMe.text="LOL you clicked me!" }


        var timesClicked = 0
        btnClickMe.setOnClickListener {
            timesClicked += 1
            textClickMe.text = timesClicked.toString()
            Toast.makeText(this, "Number Increased !", Toast.LENGTH_LONG).show()
        }

    }
}
