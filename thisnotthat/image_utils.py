import numpy as np

def bokeh_image_from_pil(pil_image):
    """
    Takes a PIL image and converts it into and RGB array for easy inclusion in a bokeh figure.
    Parameters
    ----------
    pil_image

    Returns
    -------
    a two dimensional np.array containing np.uint32 representations of each pixels RGB values.

    Examples
    --------
    import bokeh.plotting as bpl
    fig = bpl.figure()
    bokeh_image = [bokeh_image_from_pil(pil_image)]
    fig.image_rgba(bokeh_image, x=0, y=0, dw=pil_image.size[0], dh=pil_image.size[1])
    bpl.show(fig)
    """
    return np.squeeze(np.asarray(pil_image.convert("RGBA")).view(np.uint32))[::-1]