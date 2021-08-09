function drawMap(zones){
    var all_zones = JSON.parse(zones);
    for(z of all_zones){
        var element = document.getElementById(z.fields.Reference_ID);

        if (z.fields.Zone_colour == 1)
        {
            element.style = "fill: var(--green)";
        } else if(z.fields.Zone_colour == 3) {
            element.style = "fill: var(--red)";
        } else if(z.fields.Zone_colour == 2){
            element.style = "fill: var(--yellow)";
        }
    }

}

function createTooltip(key, Daily_infected, Total_infected, Active_cases, State_name) {
    var element = document.getElementById(key);
    text = "<div class='container'><div class='row'><div class='text-center p-1'><h5>" + State_name + "</h5><h5>Daily infected: " + Daily_infected + "</h5><h5>Total infected: " + Total_infected + "</h5>\n<h5>Active cases: " + Active_cases + "</h5></div></div></div>";
    element.setAttribute("title", text);

}

// map function 
document.addEventListener("DOMContentLoaded", function(event) {
    const zoomIn = $('#zoom-in');
    const zoomOut = $('#zoom-out');

    const mobilePanZoomEventHandler = {
        haltEventListeners: ['touchstart', 'touchend', 'touchmove', 'touchleave', 'touchcancel'],
        init: function (options) {
            let instance = options.instance, initialScale = 1, pannedX = 0, pannedY = 0;

            // Init Hammer
            // Listen only for pointer and touch events
            this.hammer = new Hammer(options.svgElement, {
                inputClass: Hammer.SUPPORT_POINTER_EVENTS ? Hammer.PointerEventInput : Hammer.TouchInput
            });

            // Enable pinch
            this.hammer.get('pinch').set({enable: true});

            // Handle pan
            this.hammer.on('panstart panmove', function (ev) {
                // On pan start reset panned variables
                if (ev.type === 'panstart') {
                    pannedX = 0;
                    pannedY = 0;
                }
                // Pan only the difference
                instance.panBy({x: ev.deltaX - pannedX, y: ev.deltaY - pannedY});
                pannedX = ev.deltaX;
                pannedY = ev.deltaY;
            });

            // Handle pinch
            this.hammer.on('pinchstart pinchmove', function (ev) {
                // On pinch start remember initial zoom
                if (ev.type === 'pinchstart') {
                    initialScale = instance.getZoom();
                    instance.zoomAtPoint(initialScale * ev.scale, {x: ev.center.x, y: ev.center.y}, false);
                }
                instance.zoomAtPoint(initialScale * ev.scale, {x: ev.center.x, y: ev.center.y}, false);
            })
        },
        destroy: function () {
            this.hammer.destroy();
        }
    }


    const beforePan = function (oldPan, newPan) {
        let stopHorizontal = false,
            stopVertical = false,
            gutterWidth = 100,
            gutterHeight = 300,
            sizes = this.getSizes(),
            leftLimit = -((sizes.viewBox.x + sizes.viewBox.width) * sizes.realZoom) + gutterWidth,
            rightLimit = sizes.width - gutterWidth - (sizes.viewBox.x * sizes.realZoom),
            topLimit = -((sizes.viewBox.y + sizes.viewBox.height) * sizes.realZoom) + gutterHeight,
            bottomLimit = sizes.height - gutterHeight - (sizes.viewBox.y * sizes.realZoom);

        return {
            x: Math.max(leftLimit, Math.min(rightLimit, newPan.x)),
            y: Math.max(topLimit, Math.min(bottomLimit, newPan.y))
        };
    }

    const onZoom = function (zoomLevel) {
        zoomLevel = Math.round(zoomLevel * 100 + Number.EPSILON) / 100;
        if (zoomLevel >= 8) {
            zoomIn.addClass('disabled');
            zoomOut.removeClass('disabled');
        } else if (zoomLevel <= 0.5) {
            zoomIn.removeClass('disabled');
            zoomOut.addClass('disabled');
        } else {
            zoomIn.removeClass('disabled');
            zoomOut.removeClass('disabled');
        }
    }

    const mapPanZoom = svgPanZoom('#map', {
        zoomScaleSensitivity: 0.5,
        minZoom: 0.5,
        maxZoom: 8,
        beforePan: beforePan,
        onZoom: onZoom,
        customEventsHandler: mobilePanZoomEventHandler,
       
    });

    zoomIn.click(function (event) {
        event.preventDefault();
        event.stopPropagation();
        if ($(this).hasClass('disabled')) return;
        mapPanZoom.zoomIn();
    });

    zoomOut.click(function (event) {
        event.preventDefault();
        event.stopPropagation();
        if ($(this).hasClass('disabled')) return;
        mapPanZoom.zoomOut();
    });

    $('.collapse').on('shown.bs.collapse', function () {
        scrollableList.animate({
            scrollTop: $(this).prev().offset().top - scrollableList.offset().top + scrollableList.scrollTop()
        }, 200);
    });


});
