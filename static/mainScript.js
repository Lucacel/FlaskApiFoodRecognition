$('#recogn_btn').click(function () {
   var input = $('#file');
   if (!input){
      alert('File Error 1!');
   }
   else if (!input.files){
      alert('File Error 2!');
   }
   else if (input.files.length > 1){
      alert('Please pick one file only');
   }
   else if (!input.files[0]){
      alert('File Error 3!');
   }
   else {
      alert('File picked');
   }
});

