import { Component } from '@angular/core';
import { FormBuilder } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-input-form',
  templateUrl: './input-form.component.html',
  styleUrls: ['./input-form.component.css']
})
export class InputFormComponent {
  constructor(private formbuilder:FormBuilder, private http:HttpClient) {}

  genders=[{id:1,type:'Male'},
  {id:2,type:'Female'}];

  dtypes=[{id:1,type:'0'},
  {id:2,type:'1'},
  {id:3,type:'2'},
  {id:4,type:'3'}];

  yntypes=[{id:1,type:'yes'},
  {id:2,type:'no'}]
  
  checkoutForm = this.formbuilder.group({
    age: 0,
    chestPain:'0', //0,1,2,3
    gender:'female',
    MaxHeartRate:0,
    ExerciseInducedAngina:'no', //y,n
    oldpeak:0.0,
    slope:'0', //0,1,2
    vessels:'0', // 0,1,2,3
    thalassemia:'0' // 0,1,2,3
  });
  
  
  onSubmit() {
    console.log(this.checkoutForm.value);
    this.http.post<String>('http://127.0.0.1:8000/HDP',this.checkoutForm.value).subscribe(s=>{
      if (s=='True')
       {window.alert('YOU HAVE RISK OF A HEART DISEASE')}
     else
       {window.alert('YOU ARE SAFE')}

      console.log(s)
    })
 }
}
