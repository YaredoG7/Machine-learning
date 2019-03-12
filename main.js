function move(elem, val){

    var bar = document.getElementById("elem");
    var width = 1;
    var id = setInterval(frame, 40);

    function frame (){
 
        if (width >= val){
            clearInterval(id);
        }
        else {
            width++;
            elem.style.width = width + '%';
        }

    }
}