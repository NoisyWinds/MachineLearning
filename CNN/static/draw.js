var Draw = (function () {
    var Draw = {
        init() {
            this.canvas = $('#canvas');
            this.ctx = this.canvas[0].getContext('2d');
            this.ctx.fillStyle = 'white';
            this.ctx.fillRect(0, 0, 28, 28);
            this.isDraw = false;
            this.canvas.mousedown(this.mousedown.bind(this));
            this.canvas.mouseup(this.mouseup.bind(this));
            this.canvas.mousemove(this.mousemove.bind(this));
            this.canvas.mouseleave(this.mouseup.bind(this));
            $('#upload').click(this.upload.bind(this));
            $('#clear').click(this.clear.bind(this));
        },
        mousedown(e) {
            this.isDraw = true;
            let x = e.offsetX / 300 * 28;
            let y = e.offsetY / 300 * 28;
            this.ctx.moveTo(x, y);
            this.ctx.strokeStyle = 'black';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
        },
        mouseup(e) {
            if (this.isDraw) {
                this.ctx.closePath();
                this.isDraw = false;
            }
        },
        mousemove(e) {
            if (this.isDraw) {
                let x = e.offsetX / 300 * 28;
                let y = e.offsetY / 300 * 28;
                console.log(x);
                this.ctx.lineTo(x, y);
                this.ctx.stroke();
            }
        },
        upload() {
            // 通过toDataURL的方法把画布转成base64并且与服务器进行通信
            let dataURL = this.canvas[0].toDataURL();
            console.log(dataURL);
            $.post('/', {dataURL: dataURL}, function (data) {
                alert('i think it is ' + data);
            })
        },
        clear() {
            // 填充白色来清空画布
            this.ctx.fillStyle = 'white';
            this.ctx.fillRect(0, 0, 28, 28);
        }
    };
    return Draw;
}())